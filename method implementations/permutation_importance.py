import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from quantus import (
    FaithfulnessCorrelation, Sparseness, AvgSensitivity, EffectiveComplexity,
    norm_func, perturb_func, similarity_func, baseline_replacement_by_indices,
    slic
)

# === Setup ===
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\results\\best_inception_model.keras"
IMAGE_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\glass_trash.jpg"
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# === Load model and image ===
model = load_model(MODEL_PATH)
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
x_batch = np.expand_dims(img_array, axis=0)

# === Prediction ===
preds = model.predict(x_batch)
class_index = np.argmax(preds[0])
y_batch = np.array([class_index])

# === Permutation Importance ===
def predict_fn(x): return model.predict(x)

def permutation_importance(model, image, n_permutations=50, patch_size=30):
    importance_scores = np.zeros_like(image, dtype=np.float32)
    height, width, _ = image.shape[1:]
    original_probs = predict_fn(image)[0]
    original_pred = np.argmax(original_probs)
    original_confidence = original_probs[original_pred]

    for h in range(0, height - patch_size + 1, patch_size):
        for w in range(0, width - patch_size + 1, patch_size):
            score_drop = []
            for _ in range(n_permutations):
                permuted_img = image.copy()
                permuted_img[:, h:h+patch_size, w:w+patch_size, :] = np.random.randint(
                    0, 255, (patch_size, patch_size, 3), dtype=np.uint8
                ) / 255.0
                permuted_probs = predict_fn(permuted_img)[0]
                permuted_confidence = permuted_probs[original_pred]
                score_drop.append(original_confidence - permuted_confidence)
            importance_scores[:, h:h+patch_size, w:w+patch_size, :] = np.mean(score_drop)
    return importance_scores

# === Compute Attributions ===
importance_scores = permutation_importance(model, x_batch, n_permutations=50, patch_size=30)
a_batch = np.mean(importance_scores, axis=-1, keepdims=True)

# === Segmentation (SLIC) ===
s = slic(img_array, n_segments=3)
s = s - np.min(s)
s_batch = np.expand_dims(s, axis=(0, -1))
n_segments = np.unique(s).shape[0]
safe_subset_size = min(3, n_segments)

# === Quantus Metrics Setup ===
metrics = {
    "Faithfulness": FaithfulnessCorrelation(
        nr_runs=10,
        subset_size=safe_subset_size,
        perturb_baseline="black",
        perturb_func=baseline_replacement_by_indices,
        similarity_func=similarity_func.correlation_pearson,
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Complexity": Sparseness(
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Effective Complexity": EffectiveComplexity(
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Robustness": AvgSensitivity(
        nr_samples=2,
        lower_bound=0.2,
        norm_numerator=norm_func.fro_norm,
        norm_denominator=norm_func.fro_norm,
        perturb_func=perturb_func.uniform_noise,
        similarity_func=similarity_func.difference,
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    ),
}

# === Metric Evaluation ===
results = {}

for name, metric in metrics.items():
    print(f"\nEvaluating {name} for Permutation Importance...")
    start_time = time.perf_counter()

    kwargs = {
        "model": model,
        "x_batch": x_batch,
        "y_batch": y_batch,
        "a_batch": a_batch,
    }

    if name in ["Faithfulness", "Robustness"]:
        kwargs["s_batch"] = s_batch

    if name == "Robustness":
        def explain_func(model, inputs, targets, **kwargs):
            return np.repeat(a_batch, len(inputs), axis=0)
        kwargs["explain_func"] = explain_func

    score = metric(**kwargs)
    end_time = time.perf_counter()
    print(f"{name} took {end_time - start_time:.2f} seconds.")
    results[name] = score

# === Print Results ===
print("\n--- Permutation Importance Evaluation Results ---")
for name, score in results.items():
    print(f"{name}: {score}")

normalized_scores = a_batch[0, ..., 0]
normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min() + 1e-8)

fig, axes = plt.subplots(2, 1, figsize=(5, 8))
axes[0].imshow(x_batch[0])
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(normalized_scores, cmap="hot")
axes[1].set_title("Permutation Importance Map")
axes[1].axis("off")

plt.tight_layout()
plt.show()
