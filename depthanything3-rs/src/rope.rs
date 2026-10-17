//! 2D rotary position embedding (RoPE) as upstream's `RotaryPositionEmbedding2D`: the first
//! half of each attention head's channels is rotated by the token's row, the second half by
//! its column, each half as a 1D RoPE with base frequency `BASE_FREQUENCY`.
//!
//! Positions: the camera token is at (0, 0); in blocks attending within a view, patch
//! (row, column) is at (row + 1, column + 1); in blocks attending across views every patch
//! is at (1, 1), so position only tells the camera token from the patches.

use candle_core::{DType, Device, Result, Tensor};

use crate::variant::HEAD_DIM;

const BASE_FREQUENCY: f64 = 100.0;
/// Channels rotated per axis.
const AXIS_DIM: usize = HEAD_DIM / 2;
/// Frequencies per axis (channel pairs).
const FREQUENCIES: usize = AXIS_DIM / 2;

/// cos and sin of every token's rotation angles, [1, 1, tokens, HEAD_DIM], ready to
/// broadcast over [batch, heads, tokens, HEAD_DIM].
pub struct RopeTables {
    cos: Tensor,
    sin: Tensor,
}

/// Frequency of channel pair `index` within an axis: BASE_FREQUENCY^(-index / FREQUENCIES).
fn frequency(index: usize) -> f64 {
    1.0 / BASE_FREQUENCY.powf(index as f64 / FREQUENCIES as f64)
}

/// `positions`: (row, column) per token.
pub fn rope_tables(positions: &[[u32; 2]], device: &Device, dtype: DType) -> Result<RopeTables> {
    let tokens = positions.len();
    let mut cos = vec![0f32; tokens * HEAD_DIM];
    let mut sin = vec![0f32; tokens * HEAD_DIM];
    for (token, position) in positions.iter().enumerate() {
        for (axis, &coordinate) in position.iter().enumerate() {
            for pair in 0..FREQUENCIES {
                let angle = coordinate as f64 * frequency(pair);
                // Upstream repeats the angles: channels j and j + FREQUENCIES share one.
                for repeat in 0..2 {
                    let channel = axis * AXIS_DIM + repeat * FREQUENCIES + pair;
                    cos[token * HEAD_DIM + channel] = angle.cos() as f32;
                    sin[token * HEAD_DIM + channel] = angle.sin() as f32;
                }
            }
        }
    }
    let table = |values: Vec<f32>| -> Result<Tensor> {
        Tensor::from_vec(values, (1, 1, tokens, HEAD_DIM), device)?.to_dtype(dtype)
    };
    Ok(RopeTables {
        cos: table(cos)?,
        sin: table(sin)?,
    })
}

/// Per axis half [a, b] (FREQUENCIES channels each) -> [-b, a].
fn rotate_pairs(xs: &Tensor) -> Result<Tensor> {
    let mut halves = Vec::with_capacity(4);
    for axis in 0..2 {
        let a = xs.narrow(3, axis * AXIS_DIM, FREQUENCIES)?;
        let b = xs.narrow(3, axis * AXIS_DIM + FREQUENCIES, FREQUENCIES)?;
        halves.push(b.neg()?);
        halves.push(a);
    }
    Tensor::cat(&halves, 3)
}

impl RopeTables {
    /// `xs`: [batch, heads, tokens, HEAD_DIM].
    pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        assert_eq!(
            xs.dim(2)?,
            self.cos.dim(2)?,
            "RoPE tables were built for a different token count"
        );
        xs.broadcast_mul(&self.cos)? + rotate_pairs(xs)?.broadcast_mul(&self.sin)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_the_identity() {
        let tables = rope_tables(&[[0, 0]], &Device::Cpu, DType::F32).unwrap();
        let xs = Tensor::arange(0f32, HEAD_DIM as f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 1, HEAD_DIM))
            .unwrap();
        let out = tables.apply(&xs).unwrap();
        let diff = (out - &xs).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-6);
    }

    #[test]
    fn rotation_keeps_the_norm_and_follows_the_row() {
        // Row 3, column 0: only the first half rotates; the pair (0, 16) turns by 3 rad.
        let tables = rope_tables(&[[3, 0]], &Device::Cpu, DType::F32).unwrap();
        let mut values = vec![0f32; HEAD_DIM];
        values[0] = 1.0;
        values[AXIS_DIM] = 2.0;
        let xs = Tensor::from_vec(values, (1, 1, 1, HEAD_DIM), &Device::Cpu).unwrap();
        let out = tables.apply(&xs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((out[0] - 3f32.cos()).abs() < 1e-6, "{}", out[0]);
        assert!((out[FREQUENCIES] - 3f32.sin()).abs() < 1e-6, "{}", out[FREQUENCIES]);
        assert!((out[AXIS_DIM] - 2.0).abs() < 1e-6);
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 5f32.sqrt()).abs() < 1e-5);
    }
}
