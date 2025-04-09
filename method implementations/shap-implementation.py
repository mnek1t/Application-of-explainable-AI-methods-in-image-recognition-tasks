import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shap
import quantus
# from quantus import (
#     FaithfulnessCorrelation, RelevanceRankAccuracy, Sparseness, AvgSensitivity, EffectiveComplexity, Selectivity, SensitivityN,
#     norm_func, perturb_func, similarity_func, baseline_replacement_by_indices
# )
# for i in quantus.AVAILABLE_XAI_METHODS_TF:
#     print(i)

# print(quantus.AVAILABLE_XAI_METHODS_TF)
# === Load model and image ===
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\results\\best_inception_model.keras"
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model = load_model(MODEL_PATH)

image_path = "D:\\University\\Bachelor Thesis\\xai methods\\glass_trash.jpg"
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0

x_batch = np.expand_dims(img_array, axis=0)
preds = model.predict(x_batch)
print("preds shape:", preds.shape)
print("preds[0]:", preds[0])
class_index = np.argmax(preds[0])
print("class_index:", class_index)
y_batch = np.array([class_index])

predicted_label = CLASS_NAMES[class_index]
predicted_confidence = preds[0][class_index]
print(f"\nModel Prediction: {predicted_label} ({predicted_confidence:.2f})")

# === SHAP ===
masker = shap.maskers.Image("inpaint_telea", (224, 224, 3))
explainer = shap.Explainer(model.predict, masker)
shap_values = explainer(x_batch, max_evals=500, batch_size=32)

a = shap_values.values[0][..., class_index]
a = np.sum(a, axis=-1, keepdims=True)
a_batch = np.expand_dims(a, axis=0)

# === SLIC segmentation ===
from quantus import slic
s = slic(img_array, n_segments=3)

s = s - np.min(s)
s_batch = np.expand_dims(s, axis=(0, -1))
s_channels = []
max_label = np.max(s)
for i in range(3):
    s_channel = (np.copy(s) + i * (max_label + 1)).astype(np.int32)  # обязательно int!
    s_channels.append(s_channel)

# Stack into (224, 224, 3)
s_stack = np.stack(s_channels, axis=-1)

# Expand to (1, 224, 224, 3)
s_batch_for_localisation = np.expand_dims(s_channels[0], axis=(0, -1)) 
# s_batch_for_localisation = np.expand_dims(s_stack, axis=0)
# s_batch_for_localisation = np.expand_dims(s, axis=0) 
#FOR LOCALIZATION
# s_batch = np.expand_dims(s, axis=0)
# if s_batch.ndim == 4 and s_batch.shape[-1] == 1:
#     s_batch = np.squeeze(s_batch, axis=-1)


# === Debug shapes ===
unique_segments = np.unique(s_batch)
n_segments = unique_segments.shape[0]

def explain_func(model, inputs, targets, **kwargs):
    # Compute SHAP values dynamically for the input
    masker = shap.maskers.Image("inpaint_telea", inputs.shape[1:])  # inputs is a batch
    explainer = shap.Explainer(model.predict, masker)
    shap_values = explainer(inputs, max_evals=500, batch_size=32)
    
    # Get values for each sample in batch
    attributions = []
    for i in range(len(inputs)):
        class_idx = np.argmax(model.predict(inputs[i:i+1])[0])
        shap_val = shap_values[i].values[..., class_idx]
        shap_val = np.sum(shap_val, axis=-1, keepdims=True)
        attributions.append(shap_val)
    
    return np.array(attributions)
print("s_batch unique segment labels:", unique_segments)
print("Number of superpixels (segments):", n_segments)
print(f"x_batch shape: {x_batch.shape}")
print(f"y_batch shape: {y_batch.shape}")
print(f"a_batch shape: {a_batch.shape}")
print(f"s_batch shape: {s_batch.shape}")
# print(f"s_batch_for_localisation shape: {s_batch_for_localisation.shape}")


# === Metrics Setup ===
# Safe subset_size for metrics that require superpixels
safe_subset_size = min(3, n_segments)  # or choose any value ≤ n_segments

metrics = {
    # "SensitivityN": SensitivityN(
    #     # perturb_baseline="black",
    #     # perturb_func=baseline_replacement_by_indices,
    #     # similarity_func=similarity_func.correlation_pearson,
    #     return_aggregate=True,
    #     disable_warnings=True,
    # ),
    # "Localisation": RelevanceRankAccuracy(
    #     abs=True,
    #     normalise=False,
    #     aggregate_func=np.mean,
    #     return_aggregate=True,
    #     disable_warnings=True,
    # ),
    "Robustness": quantus.AvgSensitivity(
        nr_samples=2,
        lower_bound=0.2,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference,
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Faithfulness": quantus.FaithfulnessCorrelation(
        nr_runs=10,
        subset_size=safe_subset_size,
        perturb_baseline="black",
        perturb_func=quantus.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Complexity": quantus.Sparseness(
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Effective Complexity": quantus.EffectiveComplexity(
        return_aggregate=True,
        disable_warnings=True,
    ),
    # "Selectivity": Selectivity(
    #     perturb_baseline="black",
    #     perturb_func=baseline_replacement_by_indices,
    #     abs=True,
    #     normalise=False,
    #     return_aggregate=True,
    #     disable_warnings=True
    # )
}

# === Evaluation ===
results = {}
import time
for name, metric in metrics.items():
    print(f"\nEvaluating {name} for SHAP...")
    start_time = time.perf_counter()
    kwargs = {
        "model": model,
        "x_batch": x_batch,
        "y_batch": y_batch,
        "a_batch": a_batch,
        # "s_batch": s_batch # added for localization
    }

    if name == "Localisation":
        # s_batch_for_localisation = np.expand_dims(s, axis=(0, -1))
        # s_batch_for_localisation = np.repeat(s_batch_for_localisation, 3, axis=-1)
        print("x_batch shape:", x_batch.shape)
        print("s_batch_for_localisation shape:", s_batch_for_localisation.shape)
        print("Unique labels per channel in s_batch_for_localisation:")
        print("s_batch dtype:", s_batch_for_localisation.dtype)
        print("s_batch shape:", s_batch_for_localisation.shape)
        print("x_batch dtype:", x_batch.dtype)
        # for i in range(3):
        #     print(f"Channel {i}: {np.unique(s_batch_for_localisation[0, :, :, i])}")
        print(f"s_batch_for_localisation shape: {s_batch_for_localisation.shape}")
        kwargs["s_batch"] = s_batch_for_localisation

    if name in ["Selectivity", "SensitivityN", "Faithfulness", "Robustness"]:
        kwargs["s_batch"] = s_batch

    if name == "Robustness":
        kwargs["explain_func"] = explain_func

    score = metric(**kwargs)
    end_time = time.perf_counter()
    print(f"{name} took {end_time - start_time:.2f} seconds.")
    results[name] = score

# === Results Output ===
print("\n--- Evaluation Results ---")
for name, score in results.items():
    print(f"{name}: {score}")