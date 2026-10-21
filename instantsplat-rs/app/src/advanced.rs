//! The window's "Advanced" card, the browser demo's (`../web/index.html`): the photo to
//! view from, the training steps and pose refinement (for every reconstruction and for
//! "Train N more steps"), and loading a `fit` run from its directory.

use std::path::PathBuf;

use egui::{CollapsingHeader, ComboBox, DragValue, Grid, RichText, Vec2};

use crate::App;
use crate::look::{self, FOOTER_SIZE, SMALL_SIZE};
use crate::pipeline::Orbit;

/// The most steps the field takes (a 300-step run takes seconds; this, many minutes).
const MAX_TRAIN_STEPS: usize = 100_000;
/// Column gap, row gap: the page's `.settings` grid.
const GRID_SPACING: [f32; 2] = [16.0, 10.0];
/// The dropdown's and the number field's padding: the page's `4px 6px`, a little wider.
const FIELD_PADDING: Vec2 = Vec2::new(8.0, 4.0);

impl App {
    pub(crate) fn advanced_card(&mut self, ui: &mut egui::Ui) {
        look::card(ui, |ui| {
            CollapsingHeader::new(RichText::new("Advanced").size(SMALL_SIZE).color(look::INK))
                .id_salt("advanced")
                .show_unindented(ui, |ui| {
                    ui.add_space(8.0);
                    ui.spacing_mut().button_padding = FIELD_PADDING;
                    Grid::new("advanced settings").num_columns(2).spacing(GRID_SPACING).show(ui, |ui| {
                        self.view_row(ui);
                        self.training_rows(ui);
                        self.run_row(ui);
                    });
                });
        });
    }

    fn view_row(&mut self, ui: &mut egui::Ui) {
        look::ink(ui, "View from", SMALL_SIZE);
        let mut view = self.orbit.view;
        let selected = self.photo_names.get(view).map(|name| format!("{view}: {name}")).unwrap_or_default();
        ui.add_enabled_ui(!self.busy && !self.photo_names.is_empty(), |ui| {
            ComboBox::from_id_salt("view")
                .selected_text(RichText::new(selected).size(SMALL_SIZE))
                .show_ui(ui, |ui| {
                    for (index, name) in self.photo_names.iter().enumerate() {
                        ui.selectable_value(&mut view, index, RichText::new(format!("{index}: {name}")).size(SMALL_SIZE));
                    }
                });
        });
        if view != self.orbit.view {
            self.orbit = Orbit { view, ..Orbit::default() };
            self.worker.look(self.orbit);
        }
        ui.end_row();
    }

    fn training_rows(&mut self, ui: &mut egui::Ui) {
        look::ink(ui, "Training", SMALL_SIZE);
        ui.add_enabled_ui(!self.busy, |ui| {
            ui.horizontal(|ui| {
                ui.add(DragValue::new(&mut self.train_steps).range(0..=MAX_TRAIN_STEPS));
                look::muted(ui, "steps", SMALL_SIZE);
                ui.add_space(8.0);
                ui.checkbox(&mut self.refine_poses, RichText::new("refine camera poses").size(SMALL_SIZE));
            });
        });
        ui.end_row();
        ui.label("");
        look::muted(
            ui,
            "Run on every reconstruction; 0 shows the network's Gaussians untrained.",
            FOOTER_SIZE,
        );
        ui.end_row();
        ui.label("");
        let label = format!("Train {} more steps", self.train_steps);
        if look::small_button(ui, &label, self.has_scene && !self.busy && self.train_steps > 0) {
            self.train_more();
        }
        ui.end_row();
    }

    fn run_row(&mut self, ui: &mut egui::Ui) {
        look::ink(ui, "Load a run", SMALL_SIZE);
        if look::small_button(ui, "Choose a folder…", !self.busy)
            && let Some(dir) = pick_run_dir()
        {
            self.load_run(dir);
        }
        ui.end_row();
        ui.label("");
        look::muted(ui, "A fit run: its splat.ply, cameras.json and input_view photos.", FOOTER_SIZE);
        ui.end_row();
    }
}

fn pick_run_dir() -> Option<PathBuf> {
    rfd::FileDialog::new().set_title("Choose a fit run's folder").pick_folder()
}
