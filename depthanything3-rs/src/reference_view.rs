//! Which view the backbone treats as the reference (the one whose camera frame is the
//! world, and which gets the reference camera token): with 3 or more views, upstream picks
//! it from the class tokens after block `alt_start - 2`, moves it to position 0 for the
//! rest of the backbone, and puts the outputs back in the input order.

use clap::ValueEnum;

/// Below this many views the first view is the reference, whatever the strategy.
pub const MIN_VIEWS_FOR_SELECTION: usize = 3;
/// Guard of the min-max scaling of each metric over the views.
const SCALE_EPS: f64 = 1e-8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ReferenceViewStrategy {
    /// View 0.
    First,
    /// View count / 2.
    Middle,
    /// Upstream's default: the view whose class token is the most median one in mean
    /// cosine similarity to the others, norm, and variance across channels.
    SaddleBalanced,
}

/// Each metric scaled to [0, 1] over the views.
fn min_max_scaled(values: &[f64]) -> Vec<f64> {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    values.iter().map(|v| (v - min) / (max - min + SCALE_EPS)).collect()
}

/// Unbiased variance over `values`, as `torch.var`.
fn variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

/// `class_tokens`: one row per view. Returns the reference view's index.
pub fn select_reference_view(class_tokens: &[Vec<f32>], strategy: ReferenceViewStrategy) -> usize {
    let views = class_tokens.len();
    if views < MIN_VIEWS_FOR_SELECTION {
        return 0;
    }
    match strategy {
        ReferenceViewStrategy::First => 0,
        ReferenceViewStrategy::Middle => views / 2,
        ReferenceViewStrategy::SaddleBalanced => {
            let norms: Vec<f64> = class_tokens
                .iter()
                .map(|token| token.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt())
                .collect();
            let unit: Vec<Vec<f64>> = class_tokens
                .iter()
                .zip(&norms)
                .map(|(token, norm)| token.iter().map(|&v| v as f64 / norm).collect())
                .collect();
            let mean_similarity: Vec<f64> = (0..views)
                .map(|i| {
                    (0..views)
                        .filter(|&j| j != i)
                        .map(|j| unit[i].iter().zip(&unit[j]).map(|(a, b)| a * b).sum::<f64>())
                        .sum::<f64>()
                        / (views - 1) as f64
                })
                .collect();
            let variances: Vec<f64> = unit.iter().map(|token| variance(token)).collect();
            let [similarity, norm, var] = [mean_similarity, norms, variances].map(|metric| min_max_scaled(&metric));
            let balance: Vec<f64> = (0..views)
                .map(|i| (similarity[i] - 0.5).abs() + (norm[i] - 0.5).abs() + (var[i] - 0.5).abs())
                .collect();
            // First minimum, as torch.argmin.
            let mut best = 0;
            for (i, &score) in balance.iter().enumerate() {
                if score < balance[best] {
                    best = i;
                }
            }
            best
        }
    }
}

/// The input view index at each position once `reference` is moved to the front and the
/// others keep their order: [reference, 0, .., reference - 1, reference + 1, ..].
pub fn order_with_reference_first(views: usize, reference: usize) -> Vec<usize> {
    assert!(reference < views, "reference view {reference} of {views}");
    std::iter::once(reference)
        .chain((0..views).filter(|&view| view != reference))
        .collect()
}

/// The position of each input view in `order` (the inverse permutation).
pub fn positions_of_views(order: &[usize]) -> Vec<usize> {
    let mut positions = vec![usize::MAX; order.len()];
    for (position, &view) in order.iter().enumerate() {
        positions[view] = position;
    }
    assert!(positions.iter().all(|&p| p != usize::MAX), "order is not a permutation");
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reorders_and_restores() {
        let order = order_with_reference_first(5, 2);
        assert_eq!(order, vec![2, 0, 1, 3, 4]);
        assert_eq!(positions_of_views(&order), vec![1, 2, 0, 3, 4]);
    }

    #[test]
    fn fewer_than_three_views_keep_the_first() {
        let tokens = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(select_reference_view(&tokens, ReferenceViewStrategy::Middle), 0);
    }

    #[test]
    fn saddle_balanced_picks_the_median_view() {
        // Three tokens on a line: the middle one is the most median in every metric.
        let tokens = vec![vec![1.0, 0.0, 0.0, 0.1], vec![1.0, 0.5, 0.0, 0.2], vec![1.0, 1.0, 0.0, 0.3]];
        assert_eq!(select_reference_view(&tokens, ReferenceViewStrategy::SaddleBalanced), 1);
    }
}
