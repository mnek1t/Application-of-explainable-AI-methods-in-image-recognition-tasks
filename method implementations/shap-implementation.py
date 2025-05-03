import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import shap
import quantus
import time

start_execution_time = time.perf_counter()

def generate_shap_values(model, img_tensor, class_index):
    masker = shap.maskers.Image("inpaint_telea", img_tensor.shape[1:])
    explainer = shap.Explainer(model.predict, masker, output_names=CLASS_NAMES)
    shap_values = explainer(img_tensor, max_evals=500, batch_size=32, outputs=shap.Explanation.argsort.flip[:6])
    return shap_values

def explain_func(model, inputs, targets):
    shap_values = generate_shap_values(model, inputs, targets[0]) 
    a = shap_values.values[0][..., targets[0]]
    a = np.sum(a, axis=-1, keepdims=True)
    return np.expand_dims(a, axis=0)

MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\results\\best_inception_model.keras"
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMAGE_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\bcaa-trash.jpg"

model = load_model(MODEL_PATH)

img = image.load_img(IMAGE_PATH, target_size=(224, 224))
#pixels = np.asarray(img).astype('float32')
pixels = np.asarray(img).astype('float32') / 255.0 
x_batch = np.expand_dims(pixels, axis=0)

s_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
s_batch[:, 56:168, 56:168, :] = 1
x_batch_localisation = np.transpose(x_batch, (0, 3, 1, 2)) 
single_channel_mask = s_batch[..., 0]
s_batch_localisation = np.expand_dims(single_channel_mask, axis=1)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

print("\n--- Model Prediction ---")
print(f"Predicted class index: {pred_class}")
print(f"Predicted class label: {CLASS_NAMES[pred_class]}")
print(f"Prediction confidence scores:")
for idx, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name}: {preds[0][idx]:.4f}")

shap_values = generate_shap_values(model, x_batch, pred_class)
shap.image_plot(shap_values)
a = shap_values.values[0][..., pred_class]
a = np.sum(a, axis=-1, keepdims=True)

a = np.squeeze(a, axis=-1) 
a = (a - a.min()) / (a.max() - a.min() + 1e-8)
a_batch = np.expand_dims(a, axis=0) 
a_batch = np.expand_dims(a_batch, axis=1)

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
        disable_warnings=True,
        display_progressbar=True
    ),
    "Faithfulness": quantus.FaithfulnessCorrelation(
        nr_runs=15,
        subset_size=20,
        perturb_baseline="black",
        perturb_func=quantus.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        return_aggregate=True,
        display_progressbar=True
    ),
    "Complexity": quantus.Sparseness(
        normalise=False,
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
    ),
}

results = {}
for name, metric in metrics.items():
    print(f"\nEvaluating {name} for SHAP...")
    start_time = time.perf_counter()

    kwargs = {
        "model": model,
        "x_batch": x_batch,
        "y_batch": y_batch,
        "a_batch": a_batch
    }

    if name in ["Robustness", "Randomisation"]:
        kwargs["explain_func"] = explain_func 
    if name == "Localisation":
        kwargs["x_batch"] = x_batch_localisation
        kwargs["s_batch"] = s_batch_localisation

    score = metric(**kwargs)
    end_time = time.perf_counter()
    print(f"\n{name} took {end_time - start_time:.2f} seconds.")
    results[name] = score

print("\n--- Evaluation Results ---")
for name, score in results.items():
    print(f"{name}: {score}")

end_execution_time = time.perf_counter()
print(f"\nTotal SHAP + Evaluation time: {end_execution_time - start_execution_time:.2f} seconds.")