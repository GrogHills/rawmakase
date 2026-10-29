//! Profile → Fit Camera Look: Camera Look (fitted) for the open photo, fitted to its
//! embedded JPEG off the UI thread (`camera_look::fit`, a few seconds).
use super::{Editor, history::Step, worker::Event};
use crate::camera_profiles::CameraProfile;
use crate::develop::black_white::TreatmentChoice;
use std::sync::Arc;

impl Editor {
    /// Fits the open photo's camera look; it arrives as [`Event::CameraLook`]. A fit
    /// already running for the photo is cancelled for this one.
    pub(super) fn start_camera_look(&mut self) {
        let Some(path) = self.document.path.clone() else {
            return;
        };
        let (generation, cancel) = self.document.camera_look.start();
        let id = self.load.id();
        let profile = self.document.edit.recipe().profile.clone();
        let sent = profile.clone();
        super::task::spawn(
            self.tx.clone(),
            self.context.clone(),
            move |tx| {
                let inputs = crate::camera_look::FitInputs::default();
                let result = crate::camera_look::fit(&path, &inputs, &cancel)
                    .map(|f| Box::new(f.profile))
                    .map_err(|e| e.reason);
                let _ = tx.send(Event::CameraLook {
                    id,
                    generation,
                    profile,
                    result,
                });
            },
            move |tx, message| {
                let _ = tx.send(Event::CameraLook {
                    id,
                    generation,
                    profile: sent,
                    result: Err(message),
                });
            },
        );
    }

    /// Uses a fitted look as the photo's profile, at 100%, as one History step,
    /// unless the profile changed while it was fitted (a later choice wins) or a
    /// newer fit was asked for. A refusal says why, as a failure that stays until dismissed.
    pub(super) fn camera_look_ready(
        &mut self,
        generation: u64,
        fitted_over: Option<Arc<CameraProfile>>,
        result: Result<Box<CameraProfile>, String>,
    ) {
        if generation != self.document.camera_look.id() {
            return;
        }
        self.document.camera_look.finish(generation);
        let profile = match result {
            Ok(profile) => profile,
            Err(why) => {
                self.messages.fail(format!("No camera look fitted: {why}"));
                return;
            }
        };
        if self.document.edit.recipe().profile != fitted_over {
            self.messages.warn(
                "The profile changed while the camera look was fitted; fit it again to use it",
            );
            return;
        }
        let profile = Arc::new(*profile);
        self.change_edit(Some(Step::new("Fit Camera Look", "")), |r| {
            let old = r.profile.replace(profile);
            r.profile_amount = 1.;
            // Treatment follows the profile in the same step, as a menu choice's does.
            r.follow_profile_treatment(old.as_deref(), None);
        });
        self.messages.done("Camera Look (fitted) applied");
    }

    /// Whether Profile → Fit Camera Look can run for the open photo: a RAW, decoded
    /// far enough to know it is one, with no fit already running.
    pub(super) fn can_fit_camera_look(&self) -> bool {
        self.document.path.is_some()
            && self
                .document
                .metadata
                .as_ref()
                .is_some_and(|m| !m.is_rendered())
            && !self.document.camera_look.is_running()
    }
}
