//! Proof-of-concept OpenXR/Vulkan interop for Bevy.
//!
//! Drives Bevy's multiview rendering from an OpenXR runtime: stand up an
//! `XrInstance`, let the runtime dictate which Vulkan instance and device
//! extensions Bevy must enable, create an `XrSession` against the `VkDevice`
//! Bevy ends up building, wrap the runtime's stereo swapchain as Bevy texture
//! views, and drive it all from the OpenXR frame loop.
//!
//! Per-eye poses and asymmetric projections from `xrLocateViews` are not here
//! yet, so the eyes are a fixed IPD offset and the head does not move.
//!
//! # Why this doesn't fork Bevy's renderer initialization
//!
//! OpenXR requires that specific Vulkan extensions be enabled *at instance and
//! device creation time* — the runtime names them via
//! `xrGetVulkanInstanceExtensionsKHR` / `xrGetVulkanDeviceExtensionsKHR`, and
//! session creation fails if they weren't enabled. Historically that forced XR
//! integrations to take over wgpu initialization wholesale.
//!
//! Bevy's `raw_vulkan_init` feature makes that unnecessary. It exposes
//! callbacks that fire inside wgpu's Vulkan init with mutable access to the
//! extension lists, which is exactly the hook OpenXR needs.
//!
//! # Ordering
//!
//! The callbacks have to be registered before `RenderPlugin` builds, so this is
//! split into two plugins, mirroring how DLSS does it:
//!
//! ```ignore
//! App::new()
//!     .add_plugins(OpenXrInitPlugin)   // BEFORE DefaultPlugins
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(OpenXrPlugin)
//!     .run();
//! ```
//!
//! [`OpenXrInitPlugin`] creates the `XrInstance` and registers the callbacks.
//! [`OpenXrPlugin`] creates the session in `finish`, once `RenderDevice` exists.
//!
//! # The adapter-selection gap
//!
//! OpenXR *mandates* the `VkPhysicalDevice` to render with — you don't get a
//! vote — but Bevy picks its own adapter by power preference, and the
//! `raw_vulkan_init` callbacks fire after that choice is made. There is no hook
//! in between, because `xrGetVulkanGraphicsDeviceKHR` itself needs a live
//! `VkInstance` that doesn't exist until Bevy has created one.
//!
//! Rather than guess, this checks: once the device exists, the adapter Bevy
//! chose is compared against the one OpenXR mandates, and a mismatch is
//! reported with the fix. On a single-GPU machine they always agree. On a
//! machine with a discrete and an integrated GPU they may not, and the fix is
//! to name the right adapter via `WGPU_ADAPTER_NAME` or
//! `WgpuSettings::adapter_name`.

use ash::vk::Handle as _;
use bevy::{
    camera::ManualTextureViewHandle,
    math::ops,
    prelude::*,
    render::{
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
            TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{raw_vulkan_init::RawVulkanInitSettings, RenderDevice},
        texture::{ManualTextureView, ManualTextureViews},
        RenderApp,
    },
};
use core::ffi::{c_void, CStr};
use std::error::Error;
use wgpu::hal::api::Vulkan;

/// The OpenXR instance and system, created before the renderer starts.
///
/// Absent when OpenXR is unavailable — no loader, no runtime, or no headset.
/// That is not a fatal condition: the app still runs as an ordinary Bevy app.
#[derive(Resource)]
pub struct OpenXrContext {
    /// Held because the instance's function pointers come from the loader.
    _entry: openxr::Entry,
    pub instance: openxr::Instance,
    pub system: openxr::SystemId,
}

/// A live OpenXR session bound to Bevy's Vulkan device.
///
/// Present only once [`OpenXrPlugin`] has successfully created it.
#[derive(Resource)]
pub struct OpenXrSession {
    pub session: openxr::Session<openxr::Vulkan>,
    pub frame_waiter: openxr::FrameWaiter,
    pub frame_stream: openxr::FrameStream<openxr::Vulkan>,
}

/// Why OpenXR initialization failed, carried forward so it can be reported
/// through the log once logging actually exists.
#[derive(Resource)]
struct OpenXrInitError(String);

/// The render target the XR camera draws through.
///
/// Constant for the life of the app. The [`ManualTextureView`] *behind* it is
/// swapped every frame, because `xrAcquireSwapchainImage` hands out a different
/// image each time. That works because [`ManualTextureViews`] is an
/// `ExtractResource`: the main world's copy is cloned into the render world
/// during extract, and `extract_cameras` is explicitly ordered *after* that
/// clone (`bevy_render::camera`), so a write here lands in the same frame's
/// render.
pub const XR_VIEW_HANDLE: ManualTextureViewHandle = ManualTextureViewHandle(0x5852);

/// Per-frame state for the OpenXR frame loop.
#[derive(Resource)]
pub struct XrFrameLoop {
    /// Last state reported by `XrEventDataSessionStateChanged`.
    pub state: openxr::SessionState,
    /// True between `xrBeginSession` and `xrEndSession`. Frames may only be
    /// waited on and submitted while this holds.
    pub running: bool,
    /// Set between `xrBeginFrame` and `xrEndFrame`. The presence of this is
    /// what obliges us to call `xrEndFrame`, whether or not we rendered.
    in_flight: Option<InFlightFrame>,
}

impl Default for XrFrameLoop {
    fn default() -> Self {
        Self {
            state: openxr::SessionState::UNKNOWN,
            running: false,
            in_flight: None,
        }
    }
}

/// The space the composition layer's per-eye poses are expressed in.
///
/// `xrEndFrame` will not accept a projection layer without one. `STAGE` is
/// floor-level room scale, which is what a standing scene wants; `LOCAL` is the
/// always-supported fallback, centred on wherever the headset was at startup.
#[derive(Resource)]
pub struct XrReferenceSpace(pub openxr::Space);

/// A frame that has been begun but not yet submitted.
struct InFlightFrame {
    frame_state: openxr::FrameState,
    /// Whether a swapchain image was acquired for this frame. False when the
    /// runtime said `should_render == false`, in which case there is nothing to
    /// release and no layer to submit.
    acquired: bool,
}

/// Creates the OpenXR instance and teaches Bevy's Vulkan init which extensions
/// the runtime requires.
///
/// **Must be added before `RenderPlugin`** (in practice, before
/// `DefaultPlugins`). Registering the callbacks afterwards is silently useless:
/// the instance and device are already built by then.
pub struct OpenXrInitPlugin;

impl Plugin for OpenXrInitPlugin {
    fn build(&self, app: &mut App) {
        // NOTE: this plugin has to be added before `DefaultPlugins`, which is
        // where `LogPlugin` installs the tracing subscriber -- so `info!` and
        // friends are silently discarded in here. Diagnostics go to stderr
        // directly, and the failure is also stashed in a resource so
        // `OpenXrPlugin::finish` can report it through the log once there is
        // one.
        let OpenXrInit {
            context,
            instance_extensions,
            device_extensions,
        } = match init_openxr() {
            Ok(parts) => parts,
            Err(err) => {
                eprintln!(
                    "[openxr] unavailable: {err}\n\
                     [openxr] continuing without XR; the app will run as an ordinary Bevy app."
                );
                app.insert_resource(OpenXrInitError(err.to_string()));
                return;
            }
        };

        eprintln!(
            "[openxr] instance created\n\
             [openxr]   required instance extensions ({}): {}\n\
             [openxr]   required device extensions ({}): {}",
            instance_extensions.len(),
            format_extensions(&instance_extensions),
            device_extensions.len(),
            format_extensions(&device_extensions),
        );

        {
            let mut settings = app
                .world_mut()
                .get_resource_or_init::<RawVulkanInitSettings>();

            // SAFETY: both callbacks only ever append to the extension list.
            // Nothing is removed, and every name came from the runtime's own
            // list of what it requires, so we are not asking for anything the
            // implementation doesn't support.
            unsafe {
                settings.add_create_instance_callback(move |args, _features| {
                    for extension in &instance_extensions {
                        if !args.extensions.contains(extension) {
                            args.extensions.push(extension);
                        }
                    }
                });

                settings.add_create_device_callback(move |args, _adapter, _features| {
                    for extension in &device_extensions {
                        if !args.extensions.contains(extension) {
                            args.extensions.push(extension);
                        }
                    }
                });
            }
        }

        app.insert_resource(context);
    }
}

/// Creates the [`OpenXrSession`] once Bevy's Vulkan device exists.
///
/// Add after `DefaultPlugins`. Does nothing if [`OpenXrInitPlugin`] didn't
/// manage to create an instance.
pub struct OpenXrPlugin;

impl Plugin for OpenXrPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<XrFrameLoop>();

        // The frame loop straddles the render boundary: the runtime's image has
        // to be in `ManualTextureViews` before extract, and the frame can only
        // be submitted once rendering has finished. Bevy has no main-world
        // schedule after the render sub-app runs, so a frame is submitted at
        // the top of the *next* one — `xrEndFrame(N)` still precedes
        // `xrWaitFrame(N+1)`, which is all the spec requires.
        app.add_systems(
            First,
            (xr_submit_frame, xr_poll_events, xr_begin_frame)
                .chain()
                .run_if(resource_exists::<OpenXrSession>),
        );

        // `Last` runs before extract, so the image acquired here is the one
        // `extract_cameras` resolves `XR_VIEW_HANDLE` to this frame. Acquiring
        // this late also holds the runtime's image for the shortest window, and
        // leaves room for a future `xrLocateViews` to run against the predicted
        // display time in between.
        app.add_systems(
            Last,
            xr_acquire_image.run_if(resource_exists::<OpenXrSwapchain>),
        );
    }

    // `finish` runs after `RenderPlugin` has initialized the renderer, which is
    // the earliest point a `RenderDevice` — and therefore a `VkDevice` — exists.
    fn finish(&self, app: &mut App) {
        // Re-report the init failure now that logging exists.
        if let Some(err) = app.world().get_resource::<OpenXrInitError>() {
            error!("OpenXR unavailable: {}", err.0);
            return;
        }
        if app.world().get_resource::<OpenXrContext>().is_none() {
            error!("OpenXrInitPlugin was not added before DefaultPlugins; no XR instance exists");
            return;
        }

        let session = match create_session(app) {
            Ok(session) => {
                info!("OpenXR session created against Bevy's Vulkan device");
                session
            }
            Err(err) => {
                error!("failed to create OpenXR session: {err}");
                return;
            }
        };

        match create_swapchain(app, &session.session) {
            Ok(swapchain) => {
                info!(
                    "OpenXR stereo swapchain wrapped: {} runtime-owned image(s), \
                     {}x{} x {} layer(s), {:?}",
                    swapchain.views.len(),
                    swapchain.resolution.x,
                    swapchain.resolution.y,
                    swapchain.view_count,
                    swapchain.format,
                );

                // Seed the registry so a camera targeting `XR_VIEW_HANDLE` has
                // something valid to resolve before the first acquire. Which
                // image it is doesn't matter; it is replaced every frame from
                // `xr_acquire_image`.
                if let Some(first) = swapchain.views.first().cloned() {
                    app.world_mut()
                        .resource_mut::<ManualTextureViews>()
                        .insert(XR_VIEW_HANDLE, first);
                }

                app.insert_resource(swapchain);
            }
            Err(err) => error!("failed to wrap the OpenXR swapchain: {err}"),
        }

        match create_reference_space(&session.session) {
            Ok(space) => {
                app.insert_resource(space);
            }
            Err(err) => error!("failed to create a reference space: {err}"),
        }

        app.insert_resource(session);
    }
}

/// The runtime's stereo swapchain, with each runtime-owned image wrapped as a
/// Bevy [`ManualTextureView`].
///
/// The images belong to the OpenXR runtime, not to us: they are created by
/// `xrCreateSwapchain` and handed out one at a time by
/// `xrAcquireSwapchainImage`. They are wrapped here with
/// `TextureMemory::External` and no drop guard, so wgpu will never try to free
/// memory it doesn't own.
///
/// Every image is a `D2Array` view with one layer per eye — exactly the shape
/// `Multiview` renders into.
#[derive(Resource)]
pub struct OpenXrSwapchain {
    pub swapchain: openxr::Swapchain<openxr::Vulkan>,
    /// One per runtime-owned image, in the index order
    /// `xrAcquireSwapchainImage` reports.
    pub views: Vec<ManualTextureView>,
    pub resolution: UVec2,
    pub view_count: u32,
    pub format: TextureFormat,
}

/// What [`init_openxr`] produces: the OpenXR handles, plus the Vulkan
/// extensions the runtime requires at instance and device creation.
struct OpenXrInit {
    context: OpenXrContext,
    instance_extensions: Vec<&'static CStr>,
    device_extensions: Vec<&'static CStr>,
}

/// Loads the OpenXR loader, creates an instance, and asks the runtime which
/// Vulkan extensions it requires.
///
/// The extension names are resolved here, once, rather than inside the Vulkan
/// callbacks: the callbacks run on wgpu's init path where returning an error
/// isn't possible, and doing the work up front means a runtime that refuses to
/// answer fails before Bevy has built anything.
fn init_openxr() -> Result<OpenXrInit, Box<dyn Error>> {
    let entry = load_openxr_entry()?;

    let available = entry.enumerate_extensions()?;
    if !available.khr_vulkan_enable {
        return Err("runtime does not support XR_KHR_vulkan_enable".into());
    }

    // Deliberately the v1 extension, not `khr_vulkan_enable2`. v2 has the
    // runtime create the Vulkan instance and device for you, which is the
    // opposite of what we want — Bevy creates them, and we only need to be told
    // which extensions to add.
    let mut requested = openxr::ExtensionSet::default();
    requested.khr_vulkan_enable = true;

    let instance = entry.create_instance(
        &openxr::ApplicationInfo {
            application_name: "bevy_openxr_poc",
            engine_name: "bevy",
            ..default()
        },
        &requested,
        &[],
    )?;

    let system = instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY)?;

    // The spec requires this be called before session creation, and it reports
    // the Vulkan version range the runtime supports.
    let requirements = instance.graphics_requirements::<openxr::Vulkan>(system)?;
    // stderr, not `info!` -- see the note in `OpenXrInitPlugin::build`.
    eprintln!(
        "[openxr] runtime supports Vulkan {} to {}",
        requirements.min_api_version_supported, requirements.max_api_version_supported
    );

    // The view configuration is what decides whether multiview is even the
    // right rendering strategy: a stereo config reports one entry per eye, and
    // its recommended dimensions are the size the swapchain array texture has
    // to be in step 2. Reported here because it is the first thing step 2 needs
    // and the cheapest place to catch a runtime that isn't actually stereo.
    let views = instance.enumerate_view_configuration_views(
        system,
        openxr::ViewConfigurationType::PRIMARY_STEREO,
    )?;
    eprintln!(
        "[openxr] primary stereo view configuration: {} view(s)",
        views.len()
    );
    for (index, view) in views.iter().enumerate() {
        eprintln!(
            "[openxr]   view {index}: recommended {}x{} (max {}x{}), {} sample(s)",
            view.recommended_image_rect_width,
            view.recommended_image_rect_height,
            view.max_image_rect_width,
            view.max_image_rect_height,
            view.recommended_swapchain_sample_count,
        );
    }
    if views.len() != 2 {
        eprintln!(
            "[openxr] WARNING: expected 2 views for PRIMARY_STEREO, got {}. \
             Multiview assumes one layer per eye.",
            views.len()
        );
    }

    let instance_extensions =
        parse_extension_list(&instance.vulkan_legacy_instance_extensions(system)?);
    let device_extensions =
        parse_extension_list(&instance.vulkan_legacy_device_extensions(system)?);

    Ok(OpenXrInit {
        context: OpenXrContext {
            _entry: entry,
            instance,
            system,
        },
        instance_extensions,
        device_extensions,
    })
}

/// Loads the OpenXR loader shared library.
///
/// `Entry::load()` only tries the *unversioned* `libopenxr_loader.so`, which on
/// most Linux distributions ships in the `-devel` package rather than the
/// runtime one — Fedora's `openxr-libs` installs `libopenxr_loader.so.1` and
/// nothing else, so a machine with a perfectly working OpenXR setup fails to
/// dlopen. Tools like `openxr_runtime_list` don't hit this because they linked
/// against the SONAME at build time.
///
/// So try the SONAME as well. Requiring a devel package in order to *run* an
/// application is not a reasonable thing to ask.
fn load_openxr_entry() -> Result<openxr::Entry, Box<dyn Error>> {
    #[cfg(target_os = "windows")]
    const CANDIDATES: &[&str] = &["openxr_loader.dll"];
    #[cfg(target_os = "macos")]
    const CANDIDATES: &[&str] = &["libopenxr_loader.dylib"];
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    const CANDIDATES: &[&str] = &["libopenxr_loader.so", "libopenxr_loader.so.1"];

    let mut failures = Vec::new();
    for candidate in CANDIDATES {
        // SAFETY: loading a shared library runs its initializers. These are the
        // standard OpenXR loader names, resolved through the dynamic loader's
        // normal search path.
        match unsafe { openxr::Entry::load_from(std::path::Path::new(candidate)) } {
            Ok(entry) => return Ok(entry),
            Err(err) => failures.push(format!("{candidate}: {err}")),
        }
    }

    Err(format!(
        "could not load the OpenXR loader (tried {})",
        failures.join("; ")
    )
    .into())
}

/// Vulkan formats we know how to hand to wgpu, best first.
///
/// The runtime returns its supported formats in preference order, but we can
/// only use ones wgpu has a matching [`TextureFormat`] for, so the intersection
/// is taken rather than blindly accepting the runtime's first choice. sRGB
/// variants come first: the runtime composites in sRGB, and picking a UNORM
/// format here would double-apply the transfer function.
const SWAPCHAIN_FORMATS: &[(u32, TextureFormat)] = &[
    // VK_FORMAT_R8G8B8A8_SRGB
    (43, TextureFormat::Rgba8UnormSrgb),
    // VK_FORMAT_B8G8R8A8_SRGB
    (50, TextureFormat::Bgra8UnormSrgb),
    // VK_FORMAT_R8G8B8A8_UNORM
    (37, TextureFormat::Rgba8Unorm),
    // VK_FORMAT_B8G8R8A8_UNORM
    (44, TextureFormat::Bgra8Unorm),
];

/// Creates the stereo swapchain and wraps each runtime-owned image as a
/// [`ManualTextureView`].
///
/// This is where the XR side and the multiview side meet: `array_size` is the
/// view count from the runtime's stereo view configuration, which produces
/// exactly the two-layer array texture `examples/3d/multiview.rs` builds by
/// hand.
fn create_swapchain(
    app: &App,
    session: &openxr::Session<openxr::Vulkan>,
) -> Result<OpenXrSwapchain, Box<dyn Error>> {
    let context = app.world().resource::<OpenXrContext>();

    let views = context.instance.enumerate_view_configuration_views(
        context.system,
        openxr::ViewConfigurationType::PRIMARY_STEREO,
    )?;
    let first = views.first().ok_or("runtime reported no stereo views")?;
    let resolution = UVec2::new(
        first.recommended_image_rect_width,
        first.recommended_image_rect_height,
    );
    let view_count = views.len() as u32;

    let runtime_formats = session.enumerate_swapchain_formats()?;
    let (vk_format, format) = SWAPCHAIN_FORMATS
        .iter()
        .find(|(vk, _)| runtime_formats.contains(vk))
        .copied()
        .ok_or_else(|| {
            format!(
                "no usable swapchain format; runtime offers {runtime_formats:?}, \
                 none of which map to a wgpu format we handle"
            )
        })?;

    let swapchain = session.create_swapchain(&openxr::SwapchainCreateInfo {
        create_flags: openxr::SwapchainCreateFlags::EMPTY,
        // COLOR_ATTACHMENT so Bevy can render into it; SAMPLED because the
        // compositor reads it back.
        usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT
            | openxr::SwapchainUsageFlags::SAMPLED,
        format: vk_format,
        // Single-sampled. Multiview and MSAA don't combine (WGSL has no
        // `texture_depth_multisampled_2d_array`), and the runtime recommends 1
        // anyway, so the engine constraint and the runtime's preference agree.
        sample_count: 1,
        width: resolution.x,
        height: resolution.y,
        face_count: 1,
        // One array layer per eye. This is the whole point.
        array_size: view_count,
        mip_count: 1,
    })?;

    let render_device = app
        .sub_app(RenderApp)
        .world()
        .resource::<RenderDevice>()
        .wgpu_device()
        .clone();

    let size = Extent3d {
        width: resolution.x,
        height: resolution.y,
        depth_or_array_layers: view_count,
    };

    let wrapped = swapchain
        .enumerate_images()?
        .into_iter()
        .map(|image| {
            // SAFETY: `image` is a live `VkImage` owned by the OpenXR runtime,
            // created with the dimensions, format and layer count described
            // here. `TextureMemory::External` and the absent drop callback are
            // what keep wgpu from freeing memory it does not own — the runtime
            // destroys these with the swapchain.
            let hal_texture = unsafe {
                let hal_device = render_device
                    .as_hal::<Vulkan>()
                    .ok_or("not running on the Vulkan backend")?;

                hal_device.texture_from_raw(
                    ash::vk::Image::from_raw(image),
                    &wgpu::hal::TextureDescriptor {
                        label: Some("openxr_swapchain_image"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUses::COLOR_TARGET | wgpu::TextureUses::RESOURCE,
                        memory_flags: wgpu::hal::MemoryFlags::empty(),
                        view_formats: vec![],
                    },
                    None,
                    wgpu::hal::vulkan::TextureMemory::External,
                )
            };

            // SAFETY: the descriptor matches the one above, and the image has
            // no meaningful contents until we render into it.
            let texture = unsafe {
                render_device.create_texture_from_hal::<Vulkan>(
                    hal_texture,
                    &TextureDescriptor {
                        label: Some("openxr_swapchain_image"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format,
                        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::TextureUses::UNINITIALIZED,
                )
            };

            // `D2Array` so a multiview pass can address every layer.
            let texture_view = texture.create_view(&TextureViewDescriptor {
                label: Some("openxr_swapchain_image_view"),
                dimension: Some(TextureViewDimension::D2Array),
                ..default()
            });

            Ok(ManualTextureView {
                texture_view: texture_view.into(),
                size: resolution,
                view_format: format,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok(OpenXrSwapchain {
        swapchain,
        views: wrapped,
        resolution,
        view_count,
        format,
    })
}

/// Renders an extension list for logging.
fn format_extensions(extensions: &[&'static CStr]) -> String {
    if extensions.is_empty() {
        return "(none)".to_string();
    }
    extensions
        .iter()
        .map(|extension| extension.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Splits OpenXR's space-separated extension string into `CStr`s.
///
/// wgpu wants `&'static CStr`, and these names are read once at startup and
/// live for the process, so leaking is the honest representation of that
/// lifetime rather than a workaround.
fn parse_extension_list(extensions: &str) -> Vec<&'static CStr> {
    extensions
        .split_whitespace()
        .filter_map(|name| {
            let owned = std::ffi::CString::new(name).ok()?;
            Some(&*Box::leak(owned.into_boxed_c_str()))
        })
        .collect()
}

/// Builds the Vulkan binding from Bevy's device and creates the session.
fn create_session(app: &App) -> Result<OpenXrSession, Box<dyn Error>> {
    let context = app.world().resource::<OpenXrContext>();

    let render_device = app
        .sub_app(RenderApp)
        .world()
        .resource::<RenderDevice>()
        .wgpu_device()
        .clone();

    // SAFETY: we only read raw handles out of the hal device and pass them to
    // OpenXR, which is what the Vulkan binding is defined to take. Nothing is
    // mutated and no handle outlives the device.
    let session_info = unsafe {
        let hal_device = render_device
            .as_hal::<Vulkan>()
            .ok_or("Bevy is not running on the Vulkan backend (try WGPU_BACKEND=vulkan)")?;

        let vk_instance = hal_device.shared_instance().raw_instance().handle();
        let vk_physical_device = hal_device.raw_physical_device();
        let vk_device = hal_device.raw_device().handle();

        // OpenXR dictates the physical device; Bevy chose one independently.
        // They agree on a single-GPU machine, but must be checked.
        let required_physical_device = context
            .instance
            .vulkan_graphics_device(context.system, vk_instance.as_raw() as *const c_void)?;

        if vk_physical_device.as_raw() as *const c_void != required_physical_device {
            return Err(
                "Bevy selected a different GPU than the one the OpenXR runtime requires. \
                 Set WGPU_ADAPTER_NAME (or WgpuSettings::adapter_name) to the headset's GPU."
                    .into(),
            );
        }

        openxr::vulkan::SessionCreateInfo {
            instance: vk_instance.as_raw() as *const c_void,
            physical_device: vk_physical_device.as_raw() as *const c_void,
            device: vk_device.as_raw() as *const c_void,
            queue_family_index: hal_device.queue_family_index(),
            queue_index: hal_device.queue_index(),
        }
    };

    // SAFETY: the handles in `session_info` come from the live `RenderDevice`,
    // which outlives the session for the duration of the app.
    let (session, frame_waiter, frame_stream) = unsafe {
        context
            .instance
            .create_session::<openxr::Vulkan>(context.system, &session_info)?
    };

    Ok(OpenXrSession {
        session,
        frame_waiter,
        frame_stream,
    })
}

/// Creates the reference space the composition layer's poses are relative to.
///
/// `STAGE` is preferred — it puts the origin on the floor, matching a scene
/// authored in metres with the camera at eye height. Not every runtime offers
/// it, so `LOCAL` is the fallback; every runtime is required to support that.
fn create_reference_space(
    session: &openxr::Session<openxr::Vulkan>,
) -> Result<XrReferenceSpace, Box<dyn Error>> {
    match session.create_reference_space(openxr::ReferenceSpaceType::STAGE, openxr::Posef::IDENTITY)
    {
        Ok(space) => {
            info!("using a STAGE reference space (floor-level origin)");
            Ok(XrReferenceSpace(space))
        }
        Err(stage_err) => {
            warn!("STAGE reference space unavailable ({stage_err}); falling back to LOCAL");
            let space = session.create_reference_space(
                openxr::ReferenceSpaceType::LOCAL,
                openxr::Posef::IDENTITY,
            )?;
            Ok(XrReferenceSpace(space))
        }
    }
}

/// Near plane for the per-eye projections, in metres.
///
/// Closer than Bevy's 0.1 default: in a headset your hands come well inside
/// 10cm, and reverse-Z gives up almost nothing for a near plane this small.
pub const XR_NEAR: f32 = 0.05;

/// Converts an OpenXR pose into a Bevy [`Transform`].
///
/// No axis juggling required: OpenXR and Bevy are both right-handed, Y-up,
/// -Z-forward, and both store quaternions xyzw. This is a rename, not a
/// conversion — which is worth stating explicitly, because it is exactly the
/// kind of thing that gets "fixed" with a spurious negation later.
pub fn transform_from_pose(pose: openxr::Posef) -> Transform {
    Transform {
        translation: Vec3::new(pose.position.x, pose.position.y, pose.position.z),
        rotation: Quat::from_xyzw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ),
        scale: Vec3::ONE,
    }
}

/// Builds a `clip_from_view` matrix from an OpenXR field of view.
///
/// XR runtimes report **asymmetric** per-eye FOVs — the four half-angles are
/// independent, and on most headsets the outer angle is wider than the inner
/// one. Bevy's `PerspectiveProjection` cannot express that: it takes a single
/// vertical FOV and an aspect ratio, which is symmetric by construction. So the
/// matrix is built directly.
///
/// The convention is Bevy's: right-handed view space, infinite far plane,
/// reverse-Z (near maps to 1, infinity to 0), NDC Z in `[0, 1]`. That is what
/// `bevy_math::proj` (glam's RH DirectX projections) produces, and every depth
/// comparison in Bevy's shaders assumes it. glam offers an off-centre `frustum`
/// but only with a finite far plane, so the reverse-Z form is assembled here.
pub fn clip_from_fov(fov: openxr::Fovf, near: f32) -> Mat4 {
    let tan_left = ops::tan(fov.angle_left);
    let tan_right = ops::tan(fov.angle_right);
    let tan_up = ops::tan(fov.angle_up);
    let tan_down = ops::tan(fov.angle_down);

    let tan_width = tan_right - tan_left;
    let tan_height = tan_up - tan_down;

    Mat4::from_cols(
        Vec4::new(2.0 / tan_width, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / tan_height, 0.0, 0.0),
        // The z column carries the frustum's off-centre shear. Symmetric FOVs
        // zero both terms and this collapses to the standard matrix.
        Vec4::new(
            (tan_right + tan_left) / tan_width,
            (tan_up + tan_down) / tan_height,
            0.0,
            -1.0,
        ),
        Vec4::new(0.0, 0.0, near, 0.0),
    )
}

/// The widest symmetric FOV enclosing every supplied view.
///
/// The per-eye projections live on [`MultiviewSubview`](bevy::camera::MultiviewSubview),
/// but Bevy still frustum-culls against the *camera's* own `Projection`. Left at
/// the default 45°, a camera would cull geometry the eyes can actually see, and
/// things would pop out at the periphery. This produces a symmetric projection
/// guaranteed to contain both eyes' frusta, so culling never removes anything
/// visible. It over-includes slightly, which costs a few draws and is the
/// correct direction to err.
pub fn enclosing_fov(views: &[openxr::View]) -> Option<openxr::Fovf> {
    views.iter().map(|view| view.fov).reduce(|a, b| {
        let widest = |x: f32, y: f32| if x.abs() > y.abs() { x } else { y };
        openxr::Fovf {
            angle_left: widest(a.angle_left, b.angle_left),
            angle_right: widest(a.angle_right, b.angle_right),
            angle_up: widest(a.angle_up, b.angle_up),
            angle_down: widest(a.angle_down, b.angle_down),
        }
    })
}

/// A symmetric field of view matching Bevy's default perspective projection.
///
/// A placeholder for what `xrLocateViews` reports. Real runtimes supply
/// *asymmetric* per-eye FOVs, and this and the camera's `MultiviewSubview`
/// projections are two halves of the same lie — step 3 replaces both together.
fn placeholder_fov(resolution: UVec2) -> openxr::Fovf {
    // `PerspectiveProjection::default().fov` is the vertical angle.
    let half_vertical = core::f32::consts::PI / 8.0;
    let aspect = resolution.x as f32 / resolution.y as f32;
    let half_horizontal = ops::atan(ops::tan(half_vertical) * aspect);

    openxr::Fovf {
        angle_left: -half_horizontal,
        angle_right: half_horizontal,
        angle_up: half_vertical,
        angle_down: -half_vertical,
    }
}

/// Submits the frame begun on the previous tick.
///
/// This runs at the *top* of the frame rather than after rendering because
/// there is no main-world schedule following the render sub-app. See the
/// ordering note in [`OpenXrPlugin::build`].
///
/// `xrEndFrame` is unconditional once `xrBeginFrame` has been called — even
/// with nothing rendered and no layers to show. Skipping it is the usual way
/// this loop wedges.
fn xr_submit_frame(
    mut session: ResMut<OpenXrSession>,
    swapchain: Option<ResMut<OpenXrSwapchain>>,
    space: Option<Res<XrReferenceSpace>>,
    mut frame_loop: ResMut<XrFrameLoop>,
) {
    let Some(in_flight) = frame_loop.in_flight.take() else {
        return;
    };

    let mut swapchain = swapchain.filter(|_| in_flight.acquired);

    // Release before submitting: `xrEndFrame` may only reference an image the
    // runtime owns again.
    if let Some(swapchain) = swapchain.as_mut()
        && let Err(err) = swapchain.swapchain.release_image()
    {
        error!("xrReleaseSwapchainImage failed: {err}");
    }

    // One projection view per eye, each pointing at its own array layer of the
    // single stereo swapchain image. This is where multiview pays off: both
    // eyes were rendered in one pass into one texture, and the compositor is
    // handed two slices of it.
    //
    // The poses are identity because `xrLocateViews` isn't wired up yet, so the
    // runtime will reproject as though the head never moved. The image is
    // present and wrong, which is exactly what this step is proving.
    let views: Vec<_> = match (swapchain.as_deref(), space.as_deref()) {
        (Some(swapchain), Some(_)) => (0..swapchain.view_count)
            .map(|layer| {
                openxr::CompositionLayerProjectionView::new()
                    .pose(openxr::Posef::IDENTITY)
                    .fov(placeholder_fov(swapchain.resolution))
                    .sub_image(
                        openxr::SwapchainSubImage::new()
                            .swapchain(&swapchain.swapchain)
                            .image_array_index(layer)
                            .image_rect(openxr::Rect2Di {
                                offset: openxr::Offset2Di { x: 0, y: 0 },
                                extent: openxr::Extent2Di {
                                    width: swapchain.resolution.x as i32,
                                    height: swapchain.resolution.y as i32,
                                },
                            }),
                    )
            })
            .collect(),
        _ => Vec::new(),
    };

    // `xrEndFrame` is owed unconditionally once `xrBeginFrame` was called —
    // with no layers when there is nothing to show. Skipping it is the usual
    // way this loop wedges.
    let projection;
    let submitted;
    let layers: &[&openxr::CompositionLayerBase<'_, openxr::Vulkan>] = match space.as_deref() {
        Some(space) if !views.is_empty() => {
            projection = openxr::CompositionLayerProjection::new()
                .space(&space.0)
                .views(&views);
            submitted = [&*projection];
            &submitted
        }
        _ => &[],
    };

    if let Err(err) = session.frame_stream.end(
        in_flight.frame_state.predicted_display_time,
        openxr::EnvironmentBlendMode::OPAQUE,
        layers,
    ) {
        error!("xrEndFrame failed: {err}");
    }
}

/// Drains the OpenXR event queue and drives the session state machine.
///
/// The runtime decides when the session may run: it advertises `READY`, and
/// only then may `xrBeginSession` be called and frames submitted. Nothing
/// happens at all until this transition arrives.
fn xr_poll_events(
    context: Res<OpenXrContext>,
    session: Res<OpenXrSession>,
    mut frame_loop: ResMut<XrFrameLoop>,
    mut exit: MessageWriter<AppExit>,
) {
    let mut buffer = openxr::EventDataBuffer::new();

    loop {
        let event = match context.instance.poll_event(&mut buffer) {
            Ok(Some(event)) => event,
            Ok(None) => break,
            Err(err) => {
                error!("xrPollEvent failed: {err}");
                break;
            }
        };

        match event {
            openxr::Event::SessionStateChanged(changed) => {
                let state = changed.state();
                info!(
                    "OpenXR session state: {:?} -> {:?}",
                    frame_loop.state, state
                );
                frame_loop.state = state;

                match state {
                    openxr::SessionState::READY => {
                        match session
                            .session
                            .begin(openxr::ViewConfigurationType::PRIMARY_STEREO)
                        {
                            Ok(_) => {
                                frame_loop.running = true;
                                info!("xrBeginSession OK; the frame loop is live");
                            }
                            Err(err) => error!("xrBeginSession failed: {err}"),
                        }
                    }
                    openxr::SessionState::STOPPING => {
                        frame_loop.running = false;
                        if let Err(err) = session.session.end() {
                            error!("xrEndSession failed: {err}");
                        }
                    }
                    openxr::SessionState::EXITING | openxr::SessionState::LOSS_PENDING => {
                        frame_loop.running = false;
                        exit.write(AppExit::Success);
                    }
                    _ => {}
                }
            }
            openxr::Event::InstanceLossPending(_) => {
                warn!("OpenXR instance loss pending; shutting down");
                exit.write(AppExit::Success);
            }
            openxr::Event::EventsLost(lost) => {
                warn!("OpenXR dropped {} event(s)", lost.lost_event_count());
            }
            _ => {}
        }
    }
}

/// Waits for the runtime's frame pacing, then opens a frame.
///
/// `xrWaitFrame` blocks: this is the runtime throttling the app to the
/// compositor's cadence, and it is why the whole main schedule runs after it.
fn xr_begin_frame(mut session: ResMut<OpenXrSession>, mut frame_loop: ResMut<XrFrameLoop>) {
    if !frame_loop.running {
        return;
    }

    let frame_state = match session.frame_waiter.wait() {
        Ok(frame_state) => frame_state,
        Err(err) => {
            error!("xrWaitFrame failed: {err}");
            return;
        }
    };

    if let Err(err) = session.frame_stream.begin() {
        error!("xrBeginFrame failed: {err}");
        return;
    }

    frame_loop.in_flight = Some(InFlightFrame {
        frame_state,
        acquired: false,
    });
}

/// Acquires this frame's swapchain image and points [`XR_VIEW_HANDLE`] at it.
///
/// Runs in `Last`, so the write lands before extract and the camera renders
/// into the image the runtime just handed us.
fn xr_acquire_image(
    mut swapchain: ResMut<OpenXrSwapchain>,
    mut frame_loop: ResMut<XrFrameLoop>,
    mut manual_views: ResMut<ManualTextureViews>,
) {
    let Some(in_flight) = frame_loop.in_flight.as_mut() else {
        return;
    };

    // The runtime can tell us not to bother — the session isn't visible, or the
    // compositor is throttling. We still owe it an `xrEndFrame`, but there is
    // no image and no layer.
    if !in_flight.frame_state.should_render {
        return;
    }

    let index = match swapchain.swapchain.acquire_image() {
        Ok(index) => index,
        Err(err) => {
            error!("xrAcquireSwapchainImage failed: {err}");
            return;
        }
    };

    if let Err(err) = swapchain.swapchain.wait_image(openxr::Duration::INFINITE) {
        error!("xrWaitSwapchainImage failed: {err}");
        return;
    }

    in_flight.acquired = true;

    match swapchain.views.get(index as usize) {
        Some(view) => {
            manual_views.insert(XR_VIEW_HANDLE, view.clone());
        }
        None => {
            error!(
                "runtime returned swapchain image index {index}, but only {} image(s) were wrapped",
                swapchain.views.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::proj;

    /// A symmetric FOV, expressed the way a runtime would report it.
    fn symmetric(vertical_fov: f32, aspect: f32) -> openxr::Fovf {
        let half_vertical = vertical_fov / 2.0;
        let half_horizontal = ops::atan(ops::tan(half_vertical) * aspect);
        openxr::Fovf {
            angle_left: -half_horizontal,
            angle_right: half_horizontal,
            angle_up: half_vertical,
            angle_down: -half_vertical,
        }
    }

    /// The hand-built matrix must collapse to Bevy's own for the symmetric
    /// case. This is the anchor: it pins the convention (right-handed,
    /// infinite far, reverse-Z, NDC Z in [0,1]) against the exact function
    /// `PerspectiveProjection` uses, so the asymmetric generalisation can't
    /// silently drift into a different convention.
    #[test]
    fn symmetric_fov_matches_bevys_perspective() {
        for &(fov, aspect) in &[
            (core::f32::consts::PI / 4.0, 16.0 / 9.0),
            (core::f32::consts::PI / 2.0, 896.0 / 1007.0),
            (1.7, 1.0),
        ] {
            let ours = clip_from_fov(symmetric(fov, aspect), XR_NEAR);
            let bevys = proj::perspective_infinite_reverse(fov, aspect, XR_NEAR);

            assert!(
                ours.abs_diff_eq(bevys, 1e-5),
                "fov={fov} aspect={aspect}\nours:  {ours:?}\nbevy: {bevys:?}"
            );
        }
    }

    /// Reverse-Z: a point on the near plane lands at NDC z = 1, and distance
    /// drives z toward 0. Getting this backwards renders nothing and looks
    /// like a culling bug.
    #[test]
    fn reverse_z_maps_near_to_one() {
        let clip_from_view = clip_from_fov(symmetric(core::f32::consts::PI / 2.0, 1.0), XR_NEAR);

        // View space looks down -Z, so the near plane is at z = -XR_NEAR.
        let near_point = clip_from_view * Vec4::new(0.0, 0.0, -XR_NEAR, 1.0);
        assert!((near_point.z / near_point.w - 1.0).abs() < 1e-5);

        let far_point = clip_from_view * Vec4::new(0.0, 0.0, -1000.0, 1.0);
        let ndc_z = far_point.z / far_point.w;
        assert!((0.0..0.001).contains(&ndc_z), "distant z was {ndc_z}");
    }

    /// An asymmetric FOV must put the frustum edges exactly on the NDC edges.
    /// This is the property the symmetric test cannot check, and the one that
    /// matters for a real headset: the outer half-angle is wider than the
    /// inner one, and a symmetric approximation shifts the whole image.
    #[test]
    fn asymmetric_fov_maps_frustum_edges_to_ndc_edges() {
        // Deliberately lopsided, in the direction a left eye actually reports.
        let fov = openxr::Fovf {
            angle_left: -0.95,
            angle_right: 0.75,
            angle_up: 0.8,
            angle_down: -0.9,
        };
        let clip_from_view = clip_from_fov(fov, XR_NEAR);

        // A point on the left edge of the frustum, one metre out.
        let depth = 1.0f32;
        let edge = |angle: f32| depth * ops::tan(angle);

        let left = clip_from_view * Vec4::new(edge(fov.angle_left), 0.0, -depth, 1.0);
        assert!(
            (left.x / left.w + 1.0).abs() < 1e-5,
            "left edge -> {}",
            left.x / left.w
        );

        let right = clip_from_view * Vec4::new(edge(fov.angle_right), 0.0, -depth, 1.0);
        assert!(
            (right.x / right.w - 1.0).abs() < 1e-5,
            "right edge -> {}",
            right.x / right.w
        );

        let up = clip_from_view * Vec4::new(0.0, edge(fov.angle_up), -depth, 1.0);
        assert!(
            (up.y / up.w - 1.0).abs() < 1e-5,
            "up edge -> {}",
            up.y / up.w
        );

        let down = clip_from_view * Vec4::new(0.0, edge(fov.angle_down), -depth, 1.0);
        assert!(
            (down.y / down.w + 1.0).abs() < 1e-5,
            "down edge -> {}",
            down.y / down.w
        );
    }

    /// OpenXR and Bevy share a coordinate convention, so this is a rename.
    /// The test exists to catch someone "correcting" it with a negation.
    #[test]
    fn pose_conversion_is_component_wise() {
        let pose = openxr::Posef {
            position: openxr::Vector3f {
                x: 0.032,
                y: 1.6,
                z: -0.25,
            },
            orientation: openxr::Quaternionf {
                x: 0.1,
                y: 0.2,
                z: 0.3,
                w: 0.927,
            },
        };
        let transform = transform_from_pose(pose);

        assert_eq!(transform.translation, Vec3::new(0.032, 1.6, -0.25));
        assert_eq!(transform.rotation, Quat::from_xyzw(0.1, 0.2, 0.3, 0.927));
        assert_eq!(transform.scale, Vec3::ONE);
    }

    /// The culling projection must contain both eyes, taking the wider
    /// half-angle on every side rather than either eye's own.
    #[test]
    fn enclosing_fov_takes_the_widest_of_each_side() {
        let views = [
            openxr::View {
                pose: openxr::Posef::IDENTITY,
                fov: openxr::Fovf {
                    angle_left: -0.95,
                    angle_right: 0.75,
                    angle_up: 0.8,
                    angle_down: -0.7,
                },
            },
            openxr::View {
                pose: openxr::Posef::IDENTITY,
                fov: openxr::Fovf {
                    angle_left: -0.75,
                    angle_right: 0.95,
                    angle_up: 0.7,
                    angle_down: -0.9,
                },
            },
        ];

        let fov = enclosing_fov(&views).expect("two views were supplied");
        assert_eq!(fov.angle_left, -0.95);
        assert_eq!(fov.angle_right, 0.95);
        assert_eq!(fov.angle_up, 0.8);
        assert_eq!(fov.angle_down, -0.9);
    }

    #[test]
    fn enclosing_fov_of_nothing_is_none() {
        assert!(enclosing_fov(&[]).is_none());
    }
}
