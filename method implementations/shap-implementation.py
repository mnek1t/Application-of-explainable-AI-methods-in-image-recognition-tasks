import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tensorflow.keras.models import load_model
import shap
import time
from utils.xai_constants import CLASS_NAMES, MODEL_PATH, IMAGE_PATH, TARGET_SIZE, METRICS
from utils.xai_methods import load_and_preprocess_image, preprocess_localisation_from_contours

start_execution_time = time.perf_counter()

def generate_shap_values(model, img_tensor, class_index):
    masker = shap.maskers.Image("inpaint_telea", img_tensor.shape[1:])
    explainer = shap.Explainer(model.predict, masker, output_names=CLASS_NAMES)
    shap_values = explainer(img_tensor, max_evals=1000, batch_size=50, outputs=shap.Explanation.argsort.flip[:6])
    return shap_values

def explain_func(model, inputs, targets):
    shap_values = generate_shap_values(model, inputs, targets[0]) 
    a = shap_values.values[0][..., targets[0]]
    a = np.sum(a, axis=-1, keepdims=True)
    return np.expand_dims(a, axis=0)

model = load_model(MODEL_PATH)

x_batch, img = load_and_preprocess_image(IMAGE_PATH, TARGET_SIZE)

x_batch_localisation, s_batch_localisation = preprocess_localisation_from_contours(IMAGE_PATH, TARGET_SIZE)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

shap_values = generate_shap_values(model, x_batch, pred_class)
shap.image_plot(shap_values)
a = shap_values.values[0][..., pred_class]
a = np.sum(a, axis=-1, keepdims=True)

a = np.squeeze(a, axis=-1) 
a = (a - a.min()) / (a.max() - a.min() + 1e-8)
a_batch = np.expand_dims(a, axis=0) 
a_batch = np.expand_dims(a_batch, axis=1)

results = {}
for name, metric in METRICS.items():
    print(f"\nEvaluating {name} for SHAP...")
    start_time = time.perf_counter()

    kwargs = {
        "model": model,
        "x_batch": x_batch,
        "y_batch": y_batch,
        "a_batch": a_batch
    }

    if name in ["Robustness"]:
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