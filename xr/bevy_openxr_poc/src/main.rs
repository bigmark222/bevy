//! Drives Bevy's multiview rendering from an OpenXR runtime.
//!
//! An `XrSession` created against Bevy's own `VkDevice` (step 1), the runtime's
//! stereo swapchain wrapped as Bevy texture views (step 2), per-eye poses and
//! asymmetric projections from `xrLocateViews` (step 3), and the frame loop
//! that drives them (step 4).
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
    camera::{Multiview, MultiviewSubview, RenderTarget},
    prelude::*,
    render::pipelined_rendering::PipelinedRenderingPlugin,
};
use bevy_openxr_poc::{
    clip_from_fov, placeholder_fov, OpenXrInitPlugin, OpenXrPlugin, OpenXrSwapchain, XrCamera,
    XR_NEAR, XR_VIEW_HANDLE,
};

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

    // Enough depth structure to make stereo parallax obvious: cubes at
    // increasing distance shift by visibly different amounts between the eyes.
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

    // Seed values only. `xr_locate_views` overwrites both the pose and the
    // projection of every subview each frame from `xrLocateViews`; these just
    // keep frame zero from rendering through a degenerate matrix.
    let seed = MultiviewSubview {
        view_from_camera: Transform::IDENTITY,
        clip_from_view: clip_from_fov(placeholder_fov(swapchain.resolution), XR_NEAR),
    };

    commands.spawn((
        Camera3d::default(),
        XrCamera,
        RenderTarget::TextureView(XR_VIEW_HANDLE),
        Multiview {
            views: vec![seed; swapchain.view_count as usize],
        },
        // Multiview and MSAA don't combine: WGSL has no
        // `texture_depth_multisampled_2d_array`.
        Msaa::Off,
        // The camera is the play-space origin, not the head. Eye poses arrive
        // from the runtime as subview offsets from here, already including head
        // height, so this stays at the reference space's origin — moving it is
        // how locomotion would work later.
        Transform::IDENTITY,
    ));

    info!(
        "XR camera spawned: {} view(s) into a {}x{} swapchain",
        swapchain.view_count, swapchain.resolution.x, swapchain.resolution.y,
    );
}
