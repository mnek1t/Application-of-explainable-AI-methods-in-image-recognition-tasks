import numpy as np
import time
from tensorflow.keras.models import load_model
from utils.xai_constants import MODEL_PATH, IMAGE_PATH, TARGET_SIZE, METRICS
from utils.xai_methods import load_and_preprocess_image, preprocess_localisation_from_contours, plot_results
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

model = load_model(MODEL_PATH)

x_batch, img = load_and_preprocess_image(IMAGE_PATH, TARGET_SIZE)

x_batch_localisation, s_batch_localisation = preprocess_localisation_from_contours(IMAGE_PATH, TARGET_SIZE)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

importance_scores = permutation_importance(model, x_batch, n_permutations=50, patch_size=30)
a_batch = np.mean(importance_scores, axis=-1, keepdims=True)
a_batch_complexity = np.transpose(a_batch, (0, 3, 1, 2))
normalized_scores = a_batch[0, ..., 0]
normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min() + 1e-8)
plot_results("Permutation Importance Map", normalized_scores, img, cmap="hot")

# === Metric Evaluation ===
results = {}
for name, metric in METRICS.items():
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