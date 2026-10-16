// One Adam step over the packed Gaussian parameters, with a learning rate per parameter
// slot (position, log scale, rotation, opacity, SH).

struct AdamStep {
    count: u32,
    // 1 on the first step.
    step: u32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    learning_rate_by_slot: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> first_moment: array<f32>;
@group(0) @binding(3) var<storage, read_write> second_moment: array<f32>;
@group(0) @binding(4) var<uniform> adam: AdamStep;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= adam.count {
        return;
    }
    let slot = i % PARAM_STRIDE;
    let learning_rate = adam.learning_rate_by_slot[slot / 4u][slot % 4u];
    let gradient = gradients[i];
    let m = adam.beta1 * first_moment[i] + (1.0 - adam.beta1) * gradient;
    let v = adam.beta2 * second_moment[i] + (1.0 - adam.beta2) * gradient * gradient;
    first_moment[i] = m;
    second_moment[i] = v;
    let step = f32(adam.step);
    let m_hat = m / (1.0 - pow(adam.beta1, step));
    let v_hat = v / (1.0 - pow(adam.beta2, step));
    params[i] -= learning_rate * m_hat / (sqrt(v_hat) + adam.epsilon);
}
