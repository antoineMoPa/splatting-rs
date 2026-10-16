//! Where a command's files go, and the comparison images written next to them.

use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use image::{Rgb, RgbImage};

/// The splat a run writes into its output directory.
pub const SPLAT_FILE: &str = "splat.ply";
/// The cameras a run writes next to its splat.
pub const CAMERAS_FILE: &str = "cameras.json";

/// `out`, created as a directory; refused when it is an existing file.
pub fn output_dir(out: &Path) -> Result<PathBuf> {
    ensure!(!out.is_file(), "{} is a file; --out is a directory", out.display());
    std::fs::create_dir_all(out)?;
    Ok(out.to_path_buf())
}

const PANEL_GAP: u32 = 4;
const PANEL_GAP_COLOR: Rgb<u8> = Rgb([255, 255, 255]);

/// `panels` left to right, all the same size, with a white gap between them.
pub fn side_by_side(panels: &[RgbImage]) -> RgbImage {
    assert!(!panels.is_empty(), "no panels");
    let (width, height) = (panels[0].width(), panels[0].height());
    for panel in panels {
        assert_eq!((panel.width(), panel.height()), (width, height), "panel sizes differ");
    }
    let total_width = width * panels.len() as u32 + PANEL_GAP * (panels.len() as u32 - 1);
    let mut out = RgbImage::from_pixel(total_width, height, PANEL_GAP_COLOR);
    for (index, panel) in panels.iter().enumerate() {
        image::imageops::replace(&mut out, panel, (index as u32 * (width + PANEL_GAP)) as i64, 0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panels_sit_left_to_right_with_a_gap() {
        let a = RgbImage::from_pixel(3, 2, Rgb([1, 1, 1]));
        let b = RgbImage::from_pixel(3, 2, Rgb([2, 2, 2]));
        let out = side_by_side(&[a, b]);
        assert_eq!((out.width(), out.height()), (3 + PANEL_GAP + 3, 2));
        assert_eq!(out.get_pixel(0, 0).0, [1, 1, 1]);
        assert_eq!(out.get_pixel(3, 1).0, [255, 255, 255]);
        assert_eq!(out.get_pixel(3 + PANEL_GAP, 1).0, [2, 2, 2]);
    }
}
