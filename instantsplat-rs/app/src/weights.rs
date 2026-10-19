//! The models the window offers: every Depth Anything 3 variant (`Variant::ALL`, with
//! the sizes from that table) then MapAnything; where each one's checkpoint comes from on
//! Hugging Face, and whether it is already in Hugging Face's cache
//! (`~/.cache/huggingface`, as the library crates use it).

use std::path::PathBuf;

use depthanything3_rs::variant::{Variant, WEIGHTS_FILE as DA3_WEIGHTS_FILE};
use instantsplat_rs::network::Network;
use instantsplat_rs::network_mapanything::{WEIGHTS_FILE as MAPANYTHING_WEIGHTS_FILE, WEIGHTS_REPO as MAPANYTHING_WEIGHTS_REPO};

/// Which `Network` runs each Depth Anything 3 variant.
const DA3_NETWORKS: [(Variant, Network); 4] = [
    (Variant::Small, Network::Da3Small),
    (Variant::Base, Network::Da3Base),
    (Variant::Large, Network::Da3Large),
    (Variant::Giant, Network::Da3Giant),
];
const MAPANYTHING_SIZE_LABEL: &str = "4.9 GB";

pub struct ModelChoice {
    pub network: Network,
    /// For the dropdown: "Depth Anything 3 Small (137 MB)".
    pub label: String,
    /// For the status line: "137 MB".
    pub size_label: String,
    pub repo: &'static str,
    pub file: &'static str,
}

impl ModelChoice {
    /// The checkpoint's path when Hugging Face's cache already has it.
    pub fn cached(&self) -> Option<PathBuf> {
        hf_hub::Cache::default().model(self.repo.to_string()).get(self.file)
    }
}

/// The dropdown's entries, in its order.
pub fn model_choices() -> Vec<ModelChoice> {
    let mut choices: Vec<ModelChoice> = Variant::ALL
        .iter()
        .map(|&variant| {
            let network = DA3_NETWORKS
                .iter()
                .find(|(candidate, _)| *candidate == variant)
                .map(|(_, network)| *network)
                .expect("every Depth Anything 3 variant has its Network in DA3_NETWORKS");
            let size_label = format!("{} MB", variant.weights_megabytes());
            ModelChoice {
                network,
                label: format!("{} ({size_label})", variant.label()),
                size_label,
                repo: variant.hf_repo(),
                file: DA3_WEIGHTS_FILE,
            }
        })
        .collect();
    choices.push(ModelChoice {
        network: Network::Mapanything,
        label: format!("{} ({MAPANYTHING_SIZE_LABEL})", Network::Mapanything.label()),
        size_label: MAPANYTHING_SIZE_LABEL.to_string(),
        repo: MAPANYTHING_WEIGHTS_REPO,
        file: MAPANYTHING_WEIGHTS_FILE,
    });
    choices
}

/// The entry running `network`.
pub fn model_choice(network: Network) -> ModelChoice {
    model_choices()
        .into_iter()
        .find(|choice| choice.network == network)
        .expect("every network the command line accepts is a model choice")
}
