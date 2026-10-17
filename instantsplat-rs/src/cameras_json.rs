//! The cameras the splat was optimized against, as the `cameras.json` of the 3DGS
//! reference implementation: per view, the camera center and camera-to-world rotation in
//! the splat's frame, focal lengths in pixels of the image the view was rendered at. Plus
//! `cx`, `cy` (the principal point, pixel centers at integer coordinates), which 3DGS does
//! not store and this crate needs to render the view again.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use nalgebra::{Matrix3, Vector3};
use serde_json::Value;

use crate::camera::PinholeCamera;

pub fn write_cameras_json(path: &Path, cameras: &[(String, PinholeCamera)]) -> Result<()> {
    write_cameras_json_to(BufWriter::new(File::create(path)?), cameras)
}

/// `write_cameras_json` to any writer (memory, in the browser).
pub fn write_cameras_json_to(mut out: impl Write, cameras: &[(String, PinholeCamera)]) -> Result<()> {
    writeln!(out, "[")?;
    for (index, (name, camera)) in cameras.iter().enumerate() {
        let rows: Vec<String> = (0..3)
            .map(|r| {
                format!(
                    "[{}, {}, {}]",
                    camera.rotation[(r, 0)],
                    camera.rotation[(r, 1)],
                    camera.rotation[(r, 2)]
                )
            })
            .collect();
        writeln!(
            out,
            "  {{\"id\": {index}, \"img_name\": \"{name}\", \"width\": {}, \"height\": {}, \"position\": [{}, {}, {}], \"rotation\": [{}], \"fx\": {}, \"fy\": {}, \"cx\": {}, \"cy\": {}}}{}",
            camera.width,
            camera.height,
            camera.translation.x,
            camera.translation.y,
            camera.translation.z,
            rows.join(", "),
            camera.focal[0],
            camera.focal[1],
            camera.principal_point[0],
            camera.principal_point[1],
            if index + 1 < cameras.len() { "," } else { "" }
        )?;
    }
    writeln!(out, "]")?;
    out.flush()?;
    Ok(())
}

/// Reads a `cameras.json` written by `write_cameras_json`. Files without `cx`/`cy` (older
/// runs, other tools) are refused: the principal point cannot be guessed.
pub fn read_cameras_json(path: &Path) -> Result<Vec<(String, PinholeCamera)>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse_cameras_json(&text, &path.display().to_string())
}

/// `read_cameras_json` from the file's text (a fetched file in a browser); `label` names it
/// in errors.
pub fn parse_cameras_json(text: &str, label: &str) -> Result<Vec<(String, PinholeCamera)>> {
    let entries: Value = serde_json::from_str(text).with_context(|| format!("parsing {label}"))?;
    let entries = entries.as_array().with_context(|| format!("{label}: not a JSON array"))?;
    entries
        .iter()
        .map(|entry| {
            let field = |key: &str| -> Result<&Value> {
                entry
                    .get(key)
                    .with_context(|| format!("{label}: a camera has no \"{key}\" (written by an older version?)"))
            };
            let number = |key: &str| -> Result<f64> {
                field(key)?.as_f64().with_context(|| format!("{label}: \"{key}\" is not a number"))
            };
            let numbers = |value: &Value, count: usize| -> Result<Vec<f64>> {
                let values: Vec<f64> = value
                    .as_array()
                    .and_then(|values| values.iter().map(Value::as_f64).collect())
                    .with_context(|| format!("{label}: expected an array of numbers"))?;
                anyhow::ensure!(values.len() == count, "{label}: expected {count} numbers");
                Ok(values)
            };
            let name = field("img_name")?
                .as_str()
                .with_context(|| format!("{label}: \"img_name\" is not a string"))?
                .to_string();
            let position = numbers(field("position")?, 3)?;
            let rows: Vec<Vec<f64>> = field("rotation")?
                .as_array()
                .with_context(|| format!("{label}: \"rotation\" is not an array"))?
                .iter()
                .map(|row| numbers(row, 3))
                .collect::<Result<_>>()?;
            anyhow::ensure!(rows.len() == 3, "{label}: \"rotation\" needs 3 rows");
            Ok((
                name,
                PinholeCamera {
                    rotation: Matrix3::from_fn(|r, c| rows[r][c]),
                    translation: Vector3::new(position[0], position[1], position[2]),
                    focal: [number("fx")?, number("fy")?],
                    principal_point: [number("cx")?, number("cy")?],
                    width: number("width")? as u32,
                    height: number("height")? as u32,
                },
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cameras_json_round_trips() {
        let camera = PinholeCamera {
            rotation: nalgebra::Rotation3::from_euler_angles(0.1, 0.2, 0.3).into_inner(),
            translation: Vector3::new(1.5, -2.0, 3.25),
            focal: [400.0, 410.0],
            principal_point: [146.5, 258.5],
            width: 294,
            height: 518,
        };
        let dir = std::env::temp_dir().join(format!("instantsplat-cameras-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cameras.json");
        write_cameras_json(&path, &[("frame_0001.png".to_string(), camera.clone())]).unwrap();
        let back = read_cameras_json(&path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].0, "frame_0001.png");
        assert_eq!(back[0].1.rotation, camera.rotation);
        assert_eq!(back[0].1.translation, camera.translation);
        assert_eq!(back[0].1.focal, camera.focal);
        assert_eq!(back[0].1.principal_point, camera.principal_point);
        assert_eq!((back[0].1.width, back[0].1.height), (294, 518));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
