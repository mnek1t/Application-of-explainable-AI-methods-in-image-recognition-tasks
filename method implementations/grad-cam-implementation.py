import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import cv2
import time
from utils.xai_constants import MODEL_PATH, IMAGE_PATH, TARGET_SIZE, INTENSITY, METRICS
from utils.xai_methods import load_and_preprocess_image, preprocess_localisation_from_contours, plot_results

start_execution_time = time.perf_counter()

# Generate Grad-CAM heatmap for a specific class index
def generate_gradcam_heatmap(model, img_tensor, class_index, conv_layer_name="mixed10"):
    # Create a sub-model to get convolution outputs and predictions
    grad_model = Model(
        [model.inputs],
        [model.get_layer(conv_layer_name).output, model.output]
    )
    # Compute gradients of the target class prediction with respect to convolution outputs
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_tensor, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    # Weight conv layer outputs by average gradients and generate heatmap
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap

def explain_func(model, inputs, targets):
    heatmap = generate_gradcam_heatmap(model, inputs, targets[0]) 
    heatmap_resized = cv2.resize(heatmap, TARGET_SIZE)
    a_batch = np.expand_dims(heatmap_resized, axis=0)
    return a_batch

model = load_model(MODEL_PATH)

x_batch, img = load_and_preprocess_image(IMAGE_PATH, TARGET_SIZE)

x_batch_localisation, s_batch_localisation = preprocess_localisation_from_contours(IMAGE_PATH, TARGET_SIZE)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

heatmap = generate_gradcam_heatmap(model, x_batch, pred_class)
heatmap_resized = cv2.resize(heatmap, TARGET_SIZE)
a_batch = np.expand_dims(heatmap_resized, axis=0)
# Overlay heatmap on original image
raw_img = np.asarray(image.load_img(IMAGE_PATH, target_size=TARGET_SIZE)).astype('float32')
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
overlay = heatmap_colored * INTENSITY + raw_img
overlay = np.clip(overlay / 255.0, 0, 1)

plot_results("Grad-CAM Overlay", overlay, img)

results = {}
for name, metric in METRICS.items():
    print(f"\nEvaluating {name} for Grad-CAM...")
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
    print(f"\n{name} took {end_time - start_time:.2f} seconds.")
    results[name] = score

print("\n--- Evaluation Results ---")
for name, score in results.items():
    print(f"{name}: {score}")

end_execution_time = time.perf_counter()
print(f"\nGrad-CAM took {end_execution_time - start_execution_time:.2f} seconds.")