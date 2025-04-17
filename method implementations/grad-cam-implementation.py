import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import quantus
import time
start_exececution_time = time.perf_counter()
def generate_gradcam_heatmap(model, img_tensor, class_index, conv_layer_name="mixed10"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_tensor, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap

def make_explain_func(cached_heatmap):
    def explain_func(model, inputs, targets):
        return np.expand_dims(cached_heatmap, axis=0)
    return explain_func

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

CLASS_NAMES = ['glass', 'cardboard', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\results\\best_inception_model.keras"
IMAGE_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\coca_cola_trash.jpeg"
INTENSITY = 0.5

model = load_model(MODEL_PATH)

img = image.load_img(IMAGE_PATH, target_size=(224, 224))
pixels = np.asarray(img).astype('float32')
x_batch = np.expand_dims(pixels, axis=0)

s_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
s_batch[:, 56:168, 56:168, :] = 1
x_batch_localisation = np.transpose(x_batch, (0, 3, 1, 2))
single_channel_mask = s_batch[..., 0]
s_batch_localisation = np.expand_dims(single_channel_mask, axis=1)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

heatmap = generate_gradcam_heatmap(model, x_batch, pred_class)
print('heatmap', heatmap.shape)
heatmap_resized = cv2.resize(heatmap, (224, 224))
a_batch = np.expand_dims(heatmap_resized, axis=0)

raw_img = np.asarray(image.load_img(IMAGE_PATH, target_size=(224, 224))).astype('float32')
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
overlay = heatmap_colored * INTENSITY + raw_img
overlay = np.clip(overlay / 255.0, 0, 1)

plot_results("Grad-CAM Overlay", overlay, img)

explain_func = make_explain_func(heatmap_resized)

metrics = {
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
        display_progressbar=True
    ),
    "Faithfulness": quantus.FaithfulnessCorrelation(
        nr_runs=10,
        subset_size=3,
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
    "Localisation": quantus.RelevanceRankAccuracy(
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Selectivity": quantus.Selectivity(
        perturb_baseline="black",
        patch_size=16,
        perturb_func=quantus.baseline_replacement_by_indices,
        abs=True,
        normalise=False,
        return_aggregate=True,
        disable_warnings=True,
        display_progressbar=True
    ),
    "SensitivityN": quantus.SensitivityN(
        features_in_step=256,
        n_max_percentage=0.3,
        similarity_func=quantus.similarity_func.cosine,
        perturb_baseline="black",
        perturb_func=quantus.baseline_replacement_by_indices,
        return_aggregate=True,
        disable_warnings=True,
        display_progressbar=True
    )
}

results = {}
for name, metric in metrics.items():
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
print(f"\nGrad-CAM took {end_time - start_time:.2f} seconds.")