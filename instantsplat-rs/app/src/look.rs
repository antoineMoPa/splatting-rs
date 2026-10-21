//! The page's look, as the browser demo's `index.html`: its palette, the card, the
//! dashed drop zone, the progress bar, the blue primary button and the small white one.

use egui::{
    Align, Button, Color32, CornerRadius, CursorIcon, FontFamily, FontId, Frame, Layout, Margin, Pos2, Rect, Response, RichText, Sense,
    Shadow, Shape, Stroke, TextStyle, Ui, Vec2, Visuals,
};

pub const BG: Color32 = Color32::from_rgb(0xf6, 0xf5, 0xf2);
pub const CARD: Color32 = Color32::WHITE;
pub const LINE: Color32 = Color32::from_rgb(0xe2, 0xe0, 0xdb);
pub const INK: Color32 = Color32::from_rgb(0x1d, 0x1d, 0x1b);
pub const MUTED: Color32 = Color32::from_rgb(0x6b, 0x6a, 0x66);
pub const ACCENT: Color32 = Color32::from_rgb(0x2a, 0x63, 0xd4);
pub const DROP_BORDER: Color32 = Color32::from_rgb(0xc9, 0xc6, 0xbf);
pub const DROP_HOVER_BG: Color32 = Color32::from_rgb(0xee, 0xf3, 0xfd);
/// The 3D view before anything is rendered (the rasterizer's background).
pub const VIEW_BG: Color32 = Color32::from_rgb(0x80, 0x80, 0x80);

pub const HEADING_SIZE: f32 = 26.0;
pub const BODY_SIZE: f32 = 16.0;
pub const SMALL_SIZE: f32 = 14.0;
pub const FOOTER_SIZE: f32 = 13.0;
pub const DROP_TITLE_SIZE: f32 = 18.0;

pub const PAGE_MAX_WIDTH: f32 = 760.0;
pub const PAGE_MARGIN: Margin = Margin {
    left: 16,
    right: 16,
    top: 32,
    bottom: 48,
};
pub const CARD_PADDING: f32 = 20.0;
pub const CARD_RADIUS: u8 = 14;
pub const CARD_GAP: f32 = 16.0;
pub const DROP_RADIUS: f32 = 12.0;
const DROP_PADDING: f32 = 28.0;
const DROP_DASH: f32 = 6.0;
const DROP_GAP: f32 = 4.0;
pub const PROGRESS_HEIGHT: f32 = 8.0;
pub const VIEW_RADIUS: u8 = 12;
pub const VIEW_SHADOW: Shadow = Shadow {
    offset: [0, 10],
    blur: 30,
    spread: 0,
    color: Color32::from_rgba_premultiplied(0, 0, 0, 30),
};
const BUTTON_PADDING: Vec2 = Vec2::new(18.0, 10.0);
const BUTTON_RADIUS: u8 = 10;
const SMALL_BUTTON_HEIGHT: f32 = 32.0;

/// Light visuals in the page's colours and the page's type sizes.
pub fn apply(ctx: &egui::Context) {
    let mut visuals = Visuals::light();
    visuals.panel_fill = BG;
    visuals.window_fill = CARD;
    visuals.extreme_bg_color = CARD;
    visuals.selection.bg_fill = ACCENT;
    visuals.widgets.noninteractive.fg_stroke.color = INK;
    visuals.widgets.inactive.fg_stroke.color = INK;
    visuals.widgets.hovered.fg_stroke.color = INK;
    visuals.widgets.active.fg_stroke.color = INK;
    ctx.set_visuals(visuals);
    ctx.all_styles_mut(|style| {
        style.text_styles = [
            (TextStyle::Heading, FontId::new(HEADING_SIZE, FontFamily::Proportional)),
            (TextStyle::Body, FontId::new(BODY_SIZE, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(BODY_SIZE, FontFamily::Proportional)),
            (TextStyle::Small, FontId::new(SMALL_SIZE, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(SMALL_SIZE, FontFamily::Monospace)),
        ]
        .into();
        style.spacing.button_padding = BUTTON_PADDING;
        style.spacing.item_spacing = Vec2::new(8.0, 8.0);
    });
}

pub fn muted(ui: &mut Ui, text: &str, size: f32) -> Response {
    ui.label(RichText::new(text).size(size).color(MUTED))
}

pub fn ink(ui: &mut Ui, text: &str, size: f32) -> Response {
    ui.label(RichText::new(text).size(size).color(INK))
}

/// A white card with the page's border and radius, full width.
pub fn card<R>(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0, LINE))
        .corner_radius(CornerRadius::same(CARD_RADIUS))
        .inner_margin(Margin::same(CARD_PADDING as i8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add_contents(ui)
        })
        .inner
}

/// The outline of `rect` with corners of `radius`, clockwise from the top-left corner's
/// end, closed.
fn rounded_outline(rect: Rect, radius: f32) -> Vec<Pos2> {
    const STEPS_PER_CORNER: usize = 8;
    // Each corner: its center and the angle its arc starts at (clockwise, y down).
    let corners = [
        (Pos2::new(rect.max.x - radius, rect.min.y + radius), -90.0f32),
        (Pos2::new(rect.max.x - radius, rect.max.y - radius), 0.0),
        (Pos2::new(rect.min.x + radius, rect.max.y - radius), 90.0),
        (Pos2::new(rect.min.x + radius, rect.min.y + radius), 180.0),
    ];
    let mut points = Vec::with_capacity(corners.len() * (STEPS_PER_CORNER + 1) + 1);
    for (center, start_degrees) in corners {
        for step in 0..=STEPS_PER_CORNER {
            let angle = (start_degrees + 90.0 * step as f32 / STEPS_PER_CORNER as f32).to_radians();
            points.push(center + radius * Vec2::new(angle.cos(), angle.sin()));
        }
    }
    points.push(points[0]);
    points
}

/// The dashed zone's usual words.
pub const DROP_TITLE: &str = "Choose photos";
pub const DROP_SUBTITLE: &str = "or drop them here — 2 or more JPEG or PNG, 6 to 12 works well";

/// The dashed zone (`title` over `subtitle`): clicked opens the picker; `hovering_files`
/// is a drag of files over the window. Greyed out and unclickable while `enabled` is off.
pub fn drop_zone(ui: &mut Ui, title: &str, subtitle: &str, hovering_files: bool, enabled: bool) -> Response {
    let line_height = |size: f32| size * 1.5;
    let height = DROP_PADDING * 2.0 + line_height(DROP_TITLE_SIZE) + line_height(SMALL_SIZE);
    let sense = if enabled { Sense::click() } else { Sense::hover() };
    let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), sense);
    let active = enabled && (response.hovered() || hovering_files);
    let painter = ui.painter();
    painter.rect_filled(rect, DROP_RADIUS, if active { DROP_HOVER_BG } else { CARD });
    let border = if active { ACCENT } else { DROP_BORDER };
    painter.extend(Shape::dashed_line(
        &rounded_outline(rect.shrink(1.0), DROP_RADIUS - 1.0),
        Stroke::new(2.0, border),
        DROP_DASH,
        DROP_GAP,
    ));
    let title_color = if enabled { INK } else { MUTED };
    let title_y = rect.min.y + DROP_PADDING + line_height(DROP_TITLE_SIZE) / 2.0;
    painter.text(
        Pos2::new(rect.center().x, title_y),
        egui::Align2::CENTER_CENTER,
        title,
        FontId::new(DROP_TITLE_SIZE, FontFamily::Proportional),
        title_color,
    );
    painter.text(
        Pos2::new(
            rect.center().x,
            title_y + line_height(DROP_TITLE_SIZE) / 2.0 + line_height(SMALL_SIZE) / 2.0,
        ),
        egui::Align2::CENTER_CENTER,
        subtitle,
        FontId::new(SMALL_SIZE, FontFamily::Proportional),
        MUTED,
    );
    if enabled {
        response.on_hover_cursor(CursorIcon::PointingHand)
    } else {
        response
    }
}

/// The page's thin bar, `fraction` of it in the accent colour.
pub fn progress_bar(ui: &mut Ui, fraction: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), PROGRESS_HEIGHT), Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, PROGRESS_HEIGHT / 2.0, LINE);
    let filled = Rect::from_min_size(rect.min, Vec2::new(rect.width() * fraction.clamp(0.0, 1.0), rect.height()));
    painter.rect_filled(filled, PROGRESS_HEIGHT / 2.0, ACCENT);
}

/// The blue button; faded while disabled. Whether it was clicked.
pub fn primary_button(ui: &mut Ui, text: &str, enabled: bool) -> bool {
    let button = Button::new(RichText::new(text).size(BODY_SIZE).color(Color32::WHITE))
        .fill(ACCENT)
        .stroke(Stroke::new(1.0, ACCENT))
        .corner_radius(CornerRadius::same(BUTTON_RADIUS));
    ui.add_enabled(enabled, button).on_hover_cursor(CursorIcon::PointingHand).clicked()
}

/// A column centered in the page, at most `PAGE_MAX_WIDTH` wide.
pub fn page_column<R>(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
    let width = ui.available_width().min(PAGE_MAX_WIDTH);
    let side = (ui.available_width() - width) / 2.0;
    ui.with_layout(Layout::top_down(Align::Min), |ui| {
        ui.add_space(0.0);
        let mut column = ui.new_child(egui::UiBuilder::new().max_rect(Rect::from_min_size(
            ui.cursor().min + Vec2::new(side, 0.0),
            Vec2::new(width, ui.available_height()),
        )));
        let result = add_contents(&mut column);
        ui.allocate_rect(column.min_rect(), Sense::hover());
        result
    })
    .inner
}

/// The page's small white button with the line border ("Advanced"'s); faded while
/// disabled. Whether it was clicked.
pub fn small_button(ui: &mut Ui, text: &str, enabled: bool) -> bool {
    let button = Button::new(RichText::new(text).size(SMALL_SIZE).color(INK))
        .fill(CARD)
        .stroke(Stroke::new(1.0, LINE))
        .corner_radius(CornerRadius::same(BUTTON_RADIUS))
        .min_size(Vec2::new(0.0, SMALL_BUTTON_HEIGHT));
    ui.add_enabled(enabled, button).on_hover_cursor(CursorIcon::PointingHand).clicked()
}
