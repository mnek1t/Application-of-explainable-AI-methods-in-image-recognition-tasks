import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import time
from utils.xai_constants import MODEL_PATH, IMAGE_PATH, TARGET_SIZE, METRICS
from utils.xai_methods import preprocess_localisation_from_contours, plot_results
start_exececution_time = time.perf_counter()

def generate_lime_explanations(model, inputs, targets):
    def predict_fn(images):
        images = np.array(images).astype("float32") / 255.0
        return model.predict(images)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        inputs[0].astype('double'),
        classifier_fn=predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )

    label_to_explain = targets[0]
    if label_to_explain not in explanation.top_labels:
        label_to_explain = explanation.top_labels[0] 

    lime_output, lime_mask = explanation.get_image_and_mask(
        label=label_to_explain,
        positive_only=True,
        num_features=50,
        hide_rest=False
    )
    lime_mask = np.where(lime_mask == 1, 1, 0).astype('int32')
    return lime_output, lime_mask

def explain_func(model, inputs, targets):
    _, lime_mask = generate_lime_explanations(model, inputs, targets)
    return np.expand_dims(lime_mask, axis=(0, -1))

model = load_model(MODEL_PATH)

img = image.load_img(IMAGE_PATH, target_size=TARGET_SIZE)
pixels = np.asarray(img).astype('float32')
x_batch = np.expand_dims(pixels, axis=0) 

x_batch_localisation, s_batch_localisation = preprocess_localisation_from_contours(IMAGE_PATH, TARGET_SIZE)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

lime_vis_image, lime_mask = generate_lime_explanations(model, x_batch, [pred_class])
plot_results(f"LIME Explanation", mark_boundaries(lime_vis_image / 255.0, lime_mask), img)

a_batch = np.expand_dims(lime_mask, axis=(0, -1)) 

x_batch_complexity = np.transpose(x_batch, (0, 3, 1, 2))
a_batch_complexity = np.transpose(a_batch, (0, 3, 1, 2))

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

    if name in {"Complexity", "Effective Complexity"}:
        kwargs["x_batch"] = x_batch_complexity
        kwargs["a_batch"] = a_batch_complexity
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

print("\n--- Evaluation Results ---")
for name, score in results.items():
    print(f"{name}: {score}")

end_execution_time = time.perf_counter()
print(f"\nLIME took {end_execution_time - start_exececution_time:.2f} seconds.")  