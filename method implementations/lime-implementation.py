import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import quantus
import time

start_exececution_time = time.perf_counter()

def generate_lime_explanations(model, inputs, targets):
    def predict_fn(images):
        images = np.array(images)
        return model.predict(images)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        inputs[0].astype('double'),
        classifier_fn=predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )

    lime_output, lime_mask = explanation.get_image_and_mask(
        label=targets[0],
        positive_only=True,
        num_features=50,
        hide_rest=False
    )
    lime_mask = np.where(lime_mask == 1, 1, 0).astype('int32')
    return lime_output, lime_mask

def explain_func(model, inputs, targets):
    _, lime_mask = generate_lime_explanations(model, inputs, targets)
    return np.expand_dims(lime_mask, axis=(0, -1))

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

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\results\\best_inception_model.keras"
IMAGE_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\bcaa-trash.jpg"

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

lime_vis_image, lime_mask = generate_lime_explanations(model, x_batch, [pred_class])
lime_vis = mark_boundaries(lime_vis_image / 255.0, lime_mask)
plot_results(f"LIME Explanation", mark_boundaries(lime_vis_image / 255.0, lime_mask), img)

a_batch = np.expand_dims(lime_mask, axis=(0, -1)) 
s = np.stack([lime_mask.astype("int32")]*3, axis=-1) 
s_batch = np.expand_dims(s, axis=0) 

x_batch_complexity = np.transpose(x_batch, (0, 3, 1, 2))
a_batch_complexity = np.transpose(a_batch, (0, 3, 1, 2))

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
        perturb_func= quantus.baseline_replacement_by_indices,
        similarity_func= quantus.similarity_func.correlation_pearson,
        return_aggregate=True,
        display_progressbar=True
    ),
    "Complexity":  quantus.Sparseness(
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True
    ),
    "Effective Complexity":  quantus.EffectiveComplexity(
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