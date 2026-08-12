//! Runs the step-1 OpenXR interop and reports what happened.
//!
//! Success is an `XrSession` created against Bevy's own `VkDevice` — i.e. the
//! OpenXR runtime and Bevy agreed on a Vulkan instance, physical device and
//! device. Nothing is rendered to the headset yet; that needs the swapchain
//! (step 2) and the frame loop (step 4).
//!
//! ```sh
//! WGPU_BACKEND=vulkan cargo run -p bevy_openxr_poc
//! ```
//!
//! Requires an OpenXR runtime to be installed and active. On Linux that's
//! usually Monado or SteamVR; on Windows, Meta's runtime via Quest Link. With
//! no runtime present it logs that OpenXR is unavailable and runs as an
//! ordinary Bevy app, which is the expected result on macOS.

use bevy::prelude::*;
use bevy_openxr_poc::{OpenXrInitPlugin, OpenXrPlugin, OpenXrSession};

fn main() {
    App::new()
        // Must precede DefaultPlugins: this registers the Vulkan init
        // callbacks, and RenderPlugin consumes them as it builds.
        .add_plugins(OpenXrInitPlugin)
        .add_plugins(DefaultPlugins)
        .add_plugins(OpenXrPlugin)
        .add_systems(Startup, report)
        .run();
}

fn report(session: Option<Res<OpenXrSession>>) {
    match session {
        Some(_) => info!(
            "step 1 OK: OpenXR session is live on Bevy's Vulkan device. \
             Next: wrap the runtime's stereo swapchain image as a ManualTextureView."
        ),
        None => warn!(
            "step 1 did not complete: no OpenXR session. \
             See the errors above for why."
        ),
    }
}
