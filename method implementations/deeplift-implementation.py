import numpy as np
import shap
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import quantus
import os
import random
import time

start_execution_time = time.perf_counter()

CLASS_NAMES = ['glass', 'cardboard', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\results\\best_inception_model.keras"
IMAGE_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\bcaa-trash.jpg"

BACKGROUND_DIR = "D:\\University\\Bachelor Thesis\\garbadge_dataset\\background-data"

def load_background_images(folder_path, img_size=(224, 224), num_samples=50):
    background_images = []
    image_files = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(('.jpg', '.jpeg', '.png'))]
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_path in selected_files:
        img = image.load_img(img_path, target_size=img_size)
        img_array = np.array(img).astype('float32') / 255.0
        background_images.append(img_array)
        
    return np.stack(background_images)

def generate_deeplift_shap_values(model, img_tensor, background_data):
    explainer = shap.DeepExplainer((model.input, model.output), background_data)
    shap_values = explainer.shap_values(img_tensor)
    return shap_values

def explain_func(model, inputs, targets=None):
    background_data = load_background_images(BACKGROUND_DIR)
    shap_values = generate_deeplift_shap_values(model, background_data, inputs)
    pred_class = int(targets[0]) if targets is not None else np.argmax(model.predict(inputs)[0])

    shap_map = shap_values[0][:, :, :, pred_class]
    shap_heatmap = np.max(np.abs(shap_map), axis=-1)
    shap_heatmap = np.power(np.abs(shap_heatmap), 0.5)
    mask = shap_heatmap < 0.05 * np.max(shap_heatmap)
    shap_heatmap[mask] = 0
    shap_heatmap = (shap_heatmap - shap_heatmap.min()) / (shap_heatmap.max() - shap_heatmap.min() + 1e-8)

    return np.expand_dims(shap_heatmap, axis=0)

def plot_results(title, input, img):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(input)
    ax[0].axis("off")
    ax[0].set_title(title)

    ax[1].imshow(img)
    ax[1].axis("off")
    ax[1].set_title("Original Image")
    plt.tight_layout()
    plt.show()

model = load_model(MODEL_PATH)

img = image.load_img(IMAGE_PATH, target_size=(224, 224))
pixels = np.array(img).astype('float32') / 255.0
x_batch = np.expand_dims(pixels, axis=0)

s_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
s_batch[:, 56:168, 56:168, :] = 1
x_batch_localisation = np.transpose(x_batch, (0, 3, 1, 2)) 
single_channel_mask = s_batch[..., 0]
s_batch_localisation = np.expand_dims(single_channel_mask, axis=1)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])

y_batch = np.array([pred_class])
background_data = load_background_images(BACKGROUND_DIR)
shap_values = generate_deeplift_shap_values(model, x_batch, background_data)
normalized_shap_values = shap_values / np.max(np.abs(shap_values))
shap.image_plot(shap_values, x_batch)
shap_map = shap_values[0][:, :, :, pred_class]

shap_heatmap = np.max(np.abs(shap_map), axis=-1)
shap_heatmap = np.power(np.abs(shap_heatmap), 0.5) 
mask = shap_heatmap < 0.05 * np.max(shap_heatmap)
shap_heatmap[mask] = 0

shap_heatmap = (shap_heatmap - shap_heatmap.min()) / (shap_heatmap.max() - shap_heatmap.min())
shap_heatmap_uint8 = np.uint8(255 * shap_heatmap)

shap_heatmap_color = cv2.applyColorMap(shap_heatmap_uint8, cv2.COLORMAP_JET)

original_img_uint8 = np.uint8(pixels * 255)

overlay = shap_heatmap_color * 0.5 + original_img_uint8 * 0.5
overlay = np.clip(overlay / 255.0, 0, 1)
a_batch = np.expand_dims(shap_heatmap, axis=(0)) 
plot_results("DeepLIFT Overlay", overlay, img)

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
    print(f"\nEvaluating {name} for DeepLIFT...")
    start_time = time.perf_counter()
    kwargs = {
        "model": model,
        "x_batch": x_batch,
        "y_batch": y_batch,
        "a_batch": a_batch
    }
    if name == "Robustness":
        kwargs["explain_func"] = explain_func
    if name == "Localisation":
        kwargs["x_batch"] = x_batch_localisation
        kwargs["s_batch"] = s_batch_localisation
    score = metric(**kwargs)
    end_time = time.perf_counter()
    print(f"{name} took {end_time - start_time:.2f} seconds.")
    results[name] = score
    
print("\n--- Evaluation Results ---")
for name, score in results.items():
    print(f"{name}: {score}")

end_execution_time = time.perf_counter()
print(f"\nTotal DeepLIFT time execution is: {end_execution_time - start_execution_time:.2f} seconds.")