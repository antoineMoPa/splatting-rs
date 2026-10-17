//! Loading a variant's checkpoint into the model: from Hugging Face or a local file
//! (`load_model`, native), or from the file's bytes (`model_from_safetensors_bytes`, the
//! browser).

use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::model::DepthAnything3;
use crate::reference_view::ReferenceViewStrategy;
use crate::variant::Variant;

/// Every tensor of the checkpoint is under this prefix.
const CHECKPOINT_PREFIX: &str = "model";

pub struct LoadedModel {
    pub model: DepthAnything3,
    pub weights_path: PathBuf,
    pub load_time: std::time::Duration,
}

/// `weights`: a local safetensors checkpoint; downloaded from Hugging Face when None.
#[cfg(feature = "hf-hub")]
pub fn load_model(
    variant: Variant,
    strategy: ReferenceViewStrategy,
    weights: Option<&PathBuf>,
    dtype: DType,
    device: &Device,
) -> Result<LoadedModel> {
    let weights_path = match weights {
        Some(path) => path.clone(),
        None => hf_hub::api::sync::Api::new()?
            .model(variant.spec().weights_repo.to_string())
            .get(crate::variant::WEIGHTS_FILE)?,
    };
    let started = web_time::Instant::now();
    // SAFETY: the checkpoint file is not modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], dtype, device)? };
    let model = DepthAnything3::new(variant, strategy, vb.pp(CHECKPOINT_PREFIX))?;
    Ok(LoadedModel {
        model,
        weights_path,
        load_time: started.elapsed(),
    })
}

/// The model from the bytes of a variant's `model.safetensors` (fetched by a browser).
pub fn model_from_safetensors_bytes(
    variant: Variant,
    strategy: ReferenceViewStrategy,
    safetensors: Vec<u8>,
    dtype: DType,
    device: &Device,
) -> Result<DepthAnything3> {
    let vb = VarBuilder::from_buffered_safetensors(safetensors, dtype, device)?;
    Ok(DepthAnything3::new(variant, strategy, vb.pp(CHECKPOINT_PREFIX))?)
}
