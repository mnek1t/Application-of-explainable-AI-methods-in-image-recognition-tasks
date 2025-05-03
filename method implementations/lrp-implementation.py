# should be run in Python 3.9
import tensorflow as tf
import innvestigate
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import time
import quantus
start_execution_time = time.perf_counter()

CLASS_NAMES = ['glass', 'cardboard', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\best_inception_model_TF_2_14.h5"
IMAGE_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\bcaa-trash.jpg"

def generate_lrp_explanations(model, img_tensor, class_index):
    analyzer = innvestigate.create_analyzer("lrp.alpha_1_beta_0", model)  
    explanations = analyzer.analyze(img_tensor)
    return explanations

def explain_func(model, inputs, targets):
    model_wo_softmax = Model(inputs=model.input, outputs=model.layers[-2].output)
    explanations = generate_lrp_explanations(model_wo_softmax, inputs, targets[0])
    return explanations

def plot_results(title, explanation, input_image):
    vmax = np.percentile(np.abs(explanation), 99)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(input_image)
    ax[0].axis("off")
    ax[0].set_title("Original Image")

    ax[1].imshow(explanation, cmap="seismic", vmin=-vmax, vmax=vmax)
    ax[1].axis("off")
    ax[1].set_title(title)
    
    plt.tight_layout()
    plt.show()
    
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
pixels = np.asarray(img).astype('float32') / 255.0 
x_batch = np.expand_dims(pixels, axis=0)

model = load_model(MODEL_PATH)
model_wo_softmax = Model(inputs=model.input, outputs=model.layers[-2].output)
preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

s_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
s_batch[:, 56:168, 56:168, :] = 1
x_batch_localisation = np.transpose(x_batch, (0, 3, 1, 2))
single_channel_mask = s_batch[..., 0]
s_batch_localisation = np.expand_dims(single_channel_mask, axis=1)

explanations = generate_lrp_explanations(model_wo_softmax, x_batch, pred_class)
a_batch = explanations
a_batch = np.transpose(a_batch, (0, 3, 1, 2))
a_batch_complexity = np.sum(a_batch, axis=1, keepdims=True) 
explanation = explanations[0].sum(axis=-1)
plot_results('LRP alpha beta explanation', explanation, img)
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
        nr_runs=10,
        subset_size=20,
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
        similarity_func=lambda a, b, **kwargs: quantus.similarity_func.abs_difference(np.array(a), np.array(b)),
        perturb_baseline="black",
        perturb_func=quantus.baseline_replacement_by_indices,
        return_aggregate=True,
        disable_warnings=True,
        display_progressbar=True
    )
}

results = {}
for name, metric in metrics.items():
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