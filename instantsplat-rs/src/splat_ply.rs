//! Gaussian splats as the binary PLY of the 3DGS reference implementation (x, y, z, normals,
//! f_dc_*, opacity, scale_*, rot_*), SH degree 0, which Brush and SuperSplat open.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use crate::gaussians::GaussianCloud;

/// In the order written, after x, y, z, nx, ny, nz.
const PROPERTIES: [&str; 17] = [
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2",
    "rot_3",
];

pub fn write_splat_ply(path: &Path, cloud: &GaussianCloud) -> Result<()> {
    write_splat_ply_to(BufWriter::new(File::create(path)?), cloud)
}

/// `write_splat_ply` into any writer (a download in a browser).
pub fn write_splat_ply_to(mut out: impl Write, cloud: &GaussianCloud) -> Result<()> {
    let count = cloud.len();
    write!(
        out,
        "ply\nformat binary_little_endian 1.0\ncomment Exported from instantsplat-rs\nelement vertex {count}\n"
    )?;
    for property in PROPERTIES {
        writeln!(out, "property float {property}")?;
    }
    writeln!(out, "end_header")?;
    for g in 0..count {
        let mut record = [0f32; PROPERTIES.len()];
        record[0..3].copy_from_slice(&cloud.positions[g * 3..g * 3 + 3]);
        record[6..9].copy_from_slice(&cloud.sh_dc[g * 3..g * 3 + 3]);
        record[9] = cloud.opacity_logits[g];
        record[10..13].copy_from_slice(&cloud.log_scales[g * 3..g * 3 + 3]);
        record[13..17].copy_from_slice(&cloud.rotations[g * 4..g * 4 + 4]);
        for value in record {
            out.write_all(&value.to_le_bytes())?;
        }
    }
    out.flush()?;
    Ok(())
}

/// Reads a PLY written by `write_splat_ply` or any 3DGS splat with float properties; extra
/// properties (f_rest_*, ...) are skipped.
pub fn read_splat_ply(path: &Path) -> Result<GaussianCloud> {
    parse_splat_ply(BufReader::new(File::open(path)?), &path.display().to_string())
}

/// `read_splat_ply` from any reader (a fetched file in a browser); `label` names it in errors.
pub fn parse_splat_ply(mut reader: impl BufRead, label: &str) -> Result<GaussianCloud> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    ensure!(line.trim_end() == "ply", "{label}: not a PLY file");
    let mut count = None;
    let mut properties: Vec<String> = Vec::new();
    loop {
        line.clear();
        ensure!(reader.read_line(&mut line)? > 0, "{label}: header never ends");
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["format", "binary_little_endian", "1.0"] | ["comment", ..] => {}
            ["element", "vertex", n] => count = Some(n.parse::<usize>()?),
            ["element", other, _] => bail!("{label}: unsupported element {other}"),
            ["property", "float", name] => properties.push((*name).to_string()),
            ["property", kind, name] => {
                bail!("{label}: property {name} is {kind}, not float")
            }
            ["end_header"] => break,
            _ => bail!("{label}: unexpected header line {line:?}"),
        }
    }
    let count = count.context("no vertex element")?;
    let column = |name: &str| -> Result<usize> {
        properties
            .iter()
            .position(|p| p == name)
            .with_context(|| format!("{label}: no property {name}"))
    };
    let columns: Vec<usize> = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    .iter()
    .map(|name| column(name))
    .collect::<Result<_>>()?;

    let mut bytes = vec![0u8; count * properties.len() * 4];
    reader.read_exact(&mut bytes)?;
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mut cloud = GaussianCloud {
        positions: Vec::with_capacity(count * 3),
        log_scales: Vec::with_capacity(count * 3),
        rotations: Vec::with_capacity(count * 4),
        opacity_logits: Vec::with_capacity(count),
        sh_dc: Vec::with_capacity(count * 3),
    };
    for record in values.chunks_exact(properties.len()) {
        let at = |k: usize| record[columns[k]];
        cloud.positions.extend([at(0), at(1), at(2)]);
        cloud.sh_dc.extend([at(3), at(4), at(5)]);
        cloud.opacity_logits.push(at(6));
        cloud.log_scales.extend([at(7), at(8), at(9)]);
        cloud.rotations.extend([at(10), at(11), at(12), at(13)]);
    }
    Ok(cloud)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ply_round_trips() {
        let cloud = GaussianCloud {
            positions: vec![1.0, 2.0, 3.0, -4.0, 5.5, 6.25],
            log_scales: vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            rotations: vec![1.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.5, 0.5],
            opacity_logits: vec![0.1, -2.2],
            sh_dc: vec![0.7, 0.8, 0.9, -1.0, 1.1, 1.2],
        };
        let dir = std::env::temp_dir().join(format!("instantsplat-ply-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("round_trip.ply");
        write_splat_ply(&path, &cloud).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let header_end = bytes.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        assert_eq!(bytes.len(), header_end + 2 * PROPERTIES.len() * 4);
        assert!(std::str::from_utf8(&bytes[..header_end]).unwrap().contains(
            "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\nproperty float scale_0\n"
        ));
        let back = read_splat_ply(&path).unwrap();
        assert_eq!(back, cloud);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
