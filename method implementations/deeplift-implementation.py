import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import shap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import random
import time
from utils.xai_constants import MODEL_PATH, IMAGE_PATH, TARGET_SIZE, METRICS, TEST_DATASET_PATH
from utils.xai_methods import load_and_preprocess_image ,preprocess_localisation_from_contours, plot_results

start_execution_time = time.perf_counter()

def load_background_images(root_folder, img_size, num_samples):
    image_paths = [
        os.path.join(root_folder, subfolder, fname)
        for subfolder in os.listdir(root_folder)
        for fname in os.listdir(os.path.join(root_folder, subfolder))
        if fname.lower().endswith('.jpg')
    ]

    random.shuffle(image_paths)
    selected_paths = image_paths[:num_samples]

    images = [
        np.array(image.load_img(path, target_size=img_size)).astype('float32') / 255.0
        for path in selected_paths
    ]

    return np.stack(images)

def preprocess_deeplift_explanations(shap_values, class_index):
    shap_map = shap_values[0][:, :, :, class_index]
    shap_heatmap = np.max(np.abs(shap_map), axis=-1)
    shap_heatmap = np.power(np.abs(shap_heatmap), 0.5) 
    mask = shap_heatmap < 0.05 * np.max(shap_heatmap)
    shap_heatmap[mask] = 0
    shap_heatmap = shap_heatmap / (np.max(shap_heatmap) + 1e-8)

    return shap_heatmap

def generate_deeplift_shap_values(model, img_tensor, background_data):
    explainer = shap.DeepExplainer((model.input, model.output), background_data)
    shap_values = explainer.shap_values(img_tensor)
    return shap_values

def explain_func(model, inputs, targets=None):
    background_data = load_background_images(TEST_DATASET_PATH, TARGET_SIZE, 100)
    shap_values = generate_deeplift_shap_values(model, inputs, background_data)
    pred_class = int(targets[0]) if targets is not None else np.argmax(model.predict(inputs)[0])
    shap_heatmap = preprocess_deeplift_explanations(shap_values, pred_class)
    return np.expand_dims(shap_heatmap, axis=0) 
    
model = load_model(MODEL_PATH)

x_batch, img = load_and_preprocess_image(IMAGE_PATH, TARGET_SIZE)

x_batch_localisation, s_batch_localisation = preprocess_localisation_from_contours(IMAGE_PATH, TARGET_SIZE)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

background_data = load_background_images(TEST_DATASET_PATH, TARGET_SIZE, 100)
shap_values = generate_deeplift_shap_values(model, x_batch, background_data)
heatmap = preprocess_deeplift_explanations(shap_values, pred_class)

a_batch = np.expand_dims(heatmap, axis=(0)) 

plot_results("DeepLIFT Explanations", heatmap, img, cmap='seismic')

results = {}
for name, metric in METRICS.items():
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