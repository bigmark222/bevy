//! Proof-of-concept OpenXR/Vulkan interop for Bevy.
//!
//! This is step 1 of driving Bevy's multiview rendering from an OpenXR runtime:
//! stand up an `XrInstance`, let the runtime dictate which Vulkan instance and
//! device extensions Bevy must enable, and create an `XrSession` against the
//! `VkDevice` Bevy ends up building. It renders nothing yet — success is a live
//! session handle.
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
    prelude::*,
    render::{
        renderer::{raw_vulkan_init::RawVulkanInitSettings, RenderDevice},
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
    fn build(&self, _app: &mut App) {}

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

        match create_session(app) {
            Ok(session) => {
                info!("OpenXR session created against Bevy's Vulkan device");
                app.insert_resource(session);
            }
            Err(err) => error!("failed to create OpenXR session: {err}"),
        }
    }
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
