//! Drives Bevy's multiview rendering from an OpenXR runtime.
//!
//! Steps so far: an `XrSession` created against Bevy's own `VkDevice` (step 1),
//! the runtime's stereo swapchain wrapped as Bevy texture views (step 2), and
//! the frame loop that drives them (step 4). Step 3 — per-eye poses and
//! asymmetric projections from `xrLocateViews` — is still outstanding, so the
//! projections here are symmetric placeholders and the head does not move.
//!
//! ```sh
//! WGPU_BACKEND=vulkan cargo run -p bevy_openxr_poc
//! ```
//!
//! Requires an OpenXR runtime to be installed and active. On Linux that's
//! usually Monado or SteamVR; on Windows, Meta's runtime via Quest Link. With
//! no runtime present it logs that OpenXR is unavailable and runs as an
//! ordinary Bevy app, which is the expected result on macOS.

use bevy::{
    camera::{CameraProjection, Multiview, MultiviewSubview, PerspectiveProjection, RenderTarget},
    prelude::*,
    render::pipelined_rendering::PipelinedRenderingPlugin,
};
use bevy_openxr_poc::{OpenXrInitPlugin, OpenXrPlugin, OpenXrSwapchain, XR_VIEW_HANDLE};

/// Half of a 64mm interpupillary distance, in meters. A placeholder until
/// `xrLocateViews` supplies real per-eye poses in step 3.
const HALF_IPD: f32 = 0.032;

fn main() {
    App::new()
        // Must precede DefaultPlugins: this registers the Vulkan init
        // callbacks, and RenderPlugin consumes them as it builds.
        .add_plugins(OpenXrInitPlugin)
        // Pipelined rendering runs the render schedule on its own thread,
        // overlapped with the next frame's main schedule. The OpenXR frame loop
        // is a strict per-frame sequence — `openxr::Swapchain` asserts on
        // acquire/wait/release ordering — so frame N's release would race frame
        // N+1's acquire. Disabled for the POC; making the two coexist means
        // moving the whole loop into the render world.
        .add_plugins(DefaultPlugins.build().disable::<PipelinedRenderingPlugin>())
        .add_plugins(OpenXrPlugin)
        .add_systems(Startup, setup)
        .run();
}

/// Spawns the scene and the multiview camera that renders into the runtime's
/// swapchain.
///
/// Runs after `OpenXrPlugin::finish`, so [`OpenXrSwapchain`] is present iff XR
/// came up. Without it there is nothing to render into and the app is just a
/// Bevy app reporting why.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    swapchain: Option<Res<OpenXrSwapchain>>,
) {
    let Some(swapchain) = swapchain else {
        warn!("no OpenXR swapchain; nothing to render into. See the errors above for why.");
        return;
    };

    // Something with enough depth structure to make parallax obvious once step
    // 3 gives the eyes real poses.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(8.0, 8.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
    ));
    for (index, x) in [-1.0f32, 0.0, 1.0].into_iter().enumerate() {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.3, 0.3, 0.3))),
            MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
            Transform::from_xyz(x, 0.15, -0.6 - index as f32 * 0.7),
        ));
    }
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Symmetric placeholder projections, one per eye. `aspect_ratio` has to be
    // set explicitly: Bevy only maintains the camera's own `Projection`, so a
    // projection built standalone for a subview never gets fixed up and would
    // otherwise render a square frustum into a non-square target.
    let clip_from_view = PerspectiveProjection {
        aspect_ratio: swapchain.resolution.x as f32 / swapchain.resolution.y as f32,
        ..default()
    }
    .get_clip_from_view();

    let eyes = (0..swapchain.view_count)
        .map(|index| {
            // Left eye first, matching the OpenXR stereo view configuration's
            // layer order.
            let sign = if index == 0 { -1.0 } else { 1.0 };
            MultiviewSubview {
                view_from_camera: Transform::from_xyz(sign * HALF_IPD, 0.0, 0.0),
                clip_from_view,
            }
        })
        .collect();

    commands.spawn((
        Camera3d::default(),
        RenderTarget::TextureView(XR_VIEW_HANDLE),
        Multiview { views: eyes },
        // Multiview and MSAA don't combine: WGSL has no
        // `texture_depth_multisampled_2d_array`.
        Msaa::Off,
        Transform::from_xyz(0.0, 1.6, 0.0).looking_at(Vec3::new(0.0, 0.5, -2.0), Vec3::Y),
    ));

    info!(
        "XR camera spawned: {} view(s) into a {}x{} swapchain",
        swapchain.view_count, swapchain.resolution.x, swapchain.resolution.y,
    );
}
