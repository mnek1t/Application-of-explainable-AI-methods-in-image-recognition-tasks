import tensorflow as tf
import innvestigate
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model, load_model
import numpy as np
import time
from utils.xai_constants import MODEL_PATH, IMAGE_PATH, TARGET_SIZE, METRICS
from utils.xai_methods import load_and_preprocess_image, preprocess_localisation_from_contours, plot_results

start_execution_time = time.perf_counter()

def generate_lrp_explanations(model, img_tensor, class_index):
    analyzer = innvestigate.create_analyzer("lrp.alpha_1_beta_0", model)
    explanations = analyzer.analyze(img_tensor)
    return explanations

def explain_func(model, inputs, targets):
    model_wo_softmax = Model(inputs=model.input, outputs=model.layers[-2].output)
    explanations = generate_lrp_explanations(model_wo_softmax, inputs, targets[0])
    return explanations

x_batch, img = load_and_preprocess_image(IMAGE_PATH, TARGET_SIZE)

model = load_model(MODEL_PATH)
model_wo_softmax = Model(inputs=model.input, outputs=model.layers[-2].output)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

x_batch_localisation, s_batch_localisation = preprocess_localisation_from_contours(IMAGE_PATH, TARGET_SIZE)

explanations = generate_lrp_explanations(model_wo_softmax, x_batch, pred_class)
a_batch = explanations
a_batch = np.transpose(a_batch, (0, 3, 1, 2))
a_batch_complexity = np.sum(a_batch, axis=1, keepdims=True) 

explanation = explanations[0].sum(axis=-1)
vmax = np.percentile(np.abs(explanation), 99)

plot_results('LRP αβ explanation', explanation, img, cmap="seismic", vmin=-vmax, vmax=vmax)

results = {}
for name, metric in METRICS.items():
    print(f"\nEvaluating {name} for LRP...")
    start_time = time.perf_counter()
    kwargs = {
        "model": model,
        "x_batch": x_batch,
        "y_batch": y_batch,
        "a_batch": a_batch,
    }
    if name in {"Complexity", "Effective Complexity"}:
        kwargs["a_batch"] = a_batch_complexity
    if name == "Robustness":
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
print(f"\nLRP took {end_execution_time - start_execution_time:.2f} seconds.")