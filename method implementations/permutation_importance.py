import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import quantus

start_exececution_time = time.perf_counter()

def permutation_importance(model, image, n_permutations=50, patch_size=30):
    def predict_fn(images):
        images = np.array(images)
        return model.predict(images)
    
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

def explain_func(model, inputs, targets):
    importance_scores = permutation_importance(model, inputs, n_permutations=50, patch_size=30)
    return np.mean(importance_scores, axis=-1, keepdims=True)

def plot_results(title, input, img):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(input, cmap="hot")
    ax[0].axis("off")
    ax[0].set_title(title)

    ax[1].imshow(img)
    ax[1].axis("off")
    ax[1].set_title("Original Image")
    plt.tight_layout()
    plt.show()

CLASS_NAMES = ['glass', 'cardboard', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\results\\best_inception_model.keras"
IMAGE_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\bcaa-trash.jpg"

model = load_model(MODEL_PATH)

img = image.load_img(IMAGE_PATH, target_size=(224, 224))
pixels = image.img_to_array(img) / 255.0
x_batch = np.expand_dims(pixels, axis=0)

s_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
s_batch[:, 56:168, 56:168, :] = 1
x_batch_localisation = np.transpose(x_batch, (0, 3, 1, 2)) 
single_channel_mask = s_batch[..., 0]
s_batch_localisation = np.expand_dims(single_channel_mask, axis=1)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

importance_scores = permutation_importance(model, x_batch, n_permutations=50, patch_size=30)
a_batch = np.mean(importance_scores, axis=-1, keepdims=True)
a_batch_complexity = np.transpose(a_batch, (0, 3, 1, 2))
normalized_scores = a_batch[0, ..., 0]
normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min() + 1e-8)
plot_results("Permutation Importance Map", normalized_scores, img)

metrics = {
    "Robustness": quantus.AvgSensitivity(
        nr_samples=2,
        lower_bound=0.2,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference,
        abs=True,
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True,
        display_progressbar=True
    ),
    "Faithfulness": quantus.FaithfulnessCorrelation(
        nr_runs=15,
        subset_size=20,
        perturb_baseline="black",
        perturb_func=quantus.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        return_aggregate=True,
    ),
    "Complexity": quantus.Sparseness(
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True
    ),
    "Effective Complexity": quantus.EffectiveComplexity(
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True,
    ),
    "Localisation": quantus.RelevanceRankAccuracy(
        abs=True,
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True,
    ),
    "Selectivity": quantus.Selectivity(
        perturb_baseline="black",
        patch_size=16,
        perturb_func=quantus.baseline_replacement_by_indices,
        abs=True,
        normalise=True,
        return_aggregate=True,
        display_progressbar=True
    ),
    "SensitivityN": quantus.SensitivityN(
        features_in_step=256, 
        n_max_percentage=0.3, 
        similarity_func=lambda a, b, **kwargs: quantus.similarity_func.abs_difference(np.array(a), np.array(b)),
        perturb_baseline="black",
        perturb_func=quantus.baseline_replacement_by_indices,
        return_aggregate=True,
        display_progressbar=True
    )
}

results = {}
for name, metric in metrics.items():
    print(f"\nEvaluating {name} for LIME...")
    start_time = time.perf_counter()
    kwargs = {
        "model": model,
        "x_batch": x_batch,
        "y_batch": y_batch,
        "a_batch": a_batch
    }
    if name == "Selectivity":
        kwargs["a_batch"] = a_batch_complexity
    if name == "Robustness":
        kwargs["explain_func"] = explain_func
    if name == "Localisation":
        kwargs["x_batch"] = x_batch_localisation
        kwargs["s_batch"] = s_batch_localisation
    score = metric(**kwargs)
    end_time = time.perf_counter()
    print(f"{name} took {end_time - start_time:.2f} seconds.")
    results[name] = score

print("\n--- Permutation Importance Evaluation Results ---")
for name, score in results.items():
    print(f"{name}: {score}")

end_execution_time = time.perf_counter()
print(f"\nTotal Permutation Importance + Metrics time: {end_execution_time - start_exececution_time:.2f} seconds.")