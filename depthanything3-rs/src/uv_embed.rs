//! The dense head's positional embedding: each feature map gets a sinusoidal embedding of
//! its pixels' positions on the image plane added, scaled by `EMBED_RATIO`, as upstream's
//! `create_uv_grid` + `position_grid_to_embed`. Positions span the plane's diagonal: the
//! corners are at (+-aspect, +-1) / sqrt(aspect^2 + 1), shrunk by (n - 1) / n per axis so
//! that they sit at pixel centers.

use candle_core::{DType, Device, Result, Tensor};

const EMBED_RATIO: f64 = 0.1;
const BASE_FREQUENCY: f64 = 100.0;

/// Sinusoidal embedding of one coordinate into `dim` channels: `dim / 2` sines then
/// `dim / 2` cosines, at frequencies BASE_FREQUENCY^(-k / (dim / 2)).
fn sincos(coordinate: f64, dim: usize, out: &mut [f32]) {
    let half = dim / 2;
    for k in 0..half {
        let angle = coordinate / BASE_FREQUENCY.powf(k as f64 / half as f64);
        out[k] = angle.sin() as f32;
        out[half + k] = angle.cos() as f32;
    }
}

/// [1, channels, rows, columns] embedding of a `columns` x `rows` map of an image with the
/// given aspect ratio (width / height): the first half of the channels embeds the
/// horizontal position, the second half the vertical one.
pub fn uv_embedding(
    channels: usize,
    (rows, columns): (usize, usize),
    aspect_ratio: f64,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    assert!(channels % 4 == 0, "{channels} channels do not split into 2 axes of sin and cos");
    let diagonal = (aspect_ratio * aspect_ratio + 1.0).sqrt();
    let span = [aspect_ratio / diagonal, 1.0 / diagonal];
    let coordinate = |index: usize, count: usize, axis: usize| -> f64 {
        let edge = span[axis] * (count as f64 - 1.0) / count as f64;
        if count == 1 {
            -edge
        } else {
            -edge + 2.0 * edge * index as f64 / (count as f64 - 1.0)
        }
    };
    let half = channels / 2;
    let mut embedding = vec![0f32; channels * rows * columns];
    let mut buffer = vec![0f32; half];
    for row in 0..rows {
        for column in 0..columns {
            let pixel = row * columns + column;
            for (axis, position) in [coordinate(column, columns, 0), coordinate(row, rows, 1)]
                .into_iter()
                .enumerate()
            {
                sincos(position, half, &mut buffer);
                for (k, value) in buffer.iter().enumerate() {
                    embedding[(axis * half + k) * rows * columns + pixel] = value * EMBED_RATIO as f32;
                }
            }
        }
    }
    Tensor::from_vec(embedding, (1, channels, rows, columns), device)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_sit_at_the_shrunk_span() {
        // 4 x 2 map of a 2:1 image: span (2, 1) / sqrt 5, shrunk by 3/4 and 1/2.
        let embedding = uv_embedding(8, (2, 4), 2.0, &Device::Cpu, DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Channel 0 is sin(u) * 0.1 at frequency 1: the top-left pixel.
        let u = -2.0 / 5f64.sqrt() * 0.75;
        assert!((embedding[0] - (u.sin() * 0.1) as f32).abs() < 1e-6);
        // Channel 4 is sin(v) * 0.1: the bottom-left pixel (row 1, column 0).
        let v = 1.0 / 5f64.sqrt() * 0.5;
        assert!((embedding[4 * 8 + 4] - (v.sin() * 0.1) as f32).abs() < 1e-6);
    }
}
