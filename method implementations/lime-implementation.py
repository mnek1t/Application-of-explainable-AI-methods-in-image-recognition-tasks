import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
from quantus import (
    FaithfulnessCorrelation, RelevanceRankAccuracy, Sparseness, AvgSensitivity, EffectiveComplexity, Selectivity, SensitivityN,
    norm_func, perturb_func, similarity_func, baseline_replacement_by_indices
)
# Load pre-trained ResNet50 model
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\results\\best_inception_model.keras"
model = load_model(MODEL_PATH)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load and preprocess the image
img_path = "D:\\University\\Bachelor Thesis\\xai methods\\glass_trash.jpg"
img = load_img(img_path, target_size=(224, 224))  # Resize image to match ResNet50 input size
img_array = img_to_array(img)  # Convert image to numpy array

x_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension
# img_array = preprocess_input(img_array)  # Preprocess input for ResNet50

# Display the loaded image
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# Make prediction
preds = model.predict(x_batch)
print("preds shape:", preds.shape)
print("preds[0]:", preds[0])
# decoded_preds = decode_predictions(preds, top=3)[0]  # Get top 3 predictions

# Print top 3 predictions
# print("Top 3 predictions:")
# for i, (imagenet_id, label, score) in enumerate(decoded_preds):
#     print(f"{i + 1}: {label} ({score:.2f})")

class_index = np.argmax(preds[0])
y_batch = np.array([class_index])
predicted_label = CLASS_NAMES[class_index]
predicted_confidence = preds[0][class_index]
print(f"\nModel Prediction: {predicted_label} ({predicted_confidence:.2f})")
# Create LIME image explainer
explainer = lime_image.LimeImageExplainer()

# Generate explanation
explanation = explainer.explain_instance(
    img_array.astype('double'),  # важно: LIME требует float64
    model.predict, 
    top_labels=5, 
    hide_color=0, 
    num_samples=1000
)


# Visualize the explanation
lime_attr, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True, 
    num_features=50,
    hide_rest=True
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Original image
ax[0].imshow(img)
ax[0].axis("off")
ax[0].set_title("Original Image")

# LIME explanation
lime_vis = mark_boundaries(lime_attr / 255.0, mask)
ax[1].imshow(lime_vis)
ax[1].axis("off")
ax[1].set_title(f"LIME Explanation")

plt.tight_layout()
plt.show()

a = np.stack([mask.astype("float32")]*3, axis=-1)  # (224, 224, 3)
a_batch = np.expand_dims(a, axis=0) # (1, 224, 224, 3)

s = np.stack([mask.astype("int32")]*3, axis=-1)  # (224, 224, 3)
s_batch = np.expand_dims(s, axis=0) 

def explain_func(model, inputs, targets=None, **kwargs):
    explainer = lime_image.LimeImageExplainer()
    attributions = []
    for input_img in inputs:
        if input_img.ndim == 3:
            input_img = np.expand_dims(input_img, axis=0)

        explanation = explainer.explain_instance(
            input_img[0].astype('double'),
            classifier_fn=model.predict,
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )
        mask, _ = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True, 
            num_features=50,
            hide_rest=True
        )
        attr = np.stack([mask.astype('float32')] * 3, axis=-1)  # Ensure (224, 224, 3)
        attributions.append(attr)
    return np.array(attributions)


print(f"x_batch shape: {x_batch.shape}")
print(f"a_batch shape: {a_batch.shape}")
print(f"s_batch shape: {s_batch.shape}")
print(f"y_batch shape: {y_batch.shape}")

metrics = {
    # "SensitivityN": SensitivityN(
    #     return_aggregate=True,
    #     disable_warnings=True
    # ),
    # "Robustness": AvgSensitivity(
    #     nr_samples=2,
    #     lower_bound=0.2,
    #     norm_numerator=norm_func.fro_norm,
    #     norm_denominator=norm_func.fro_norm,
    #     perturb_func=perturb_func.uniform_noise,
    #     similarity_func=similarity_func.difference,
    #     abs=True,
    #     normalise=False,
    #     aggregate_func=np.mean,
    #     return_aggregate=True,
    #     disable_warnings=True,
    # ),
    "Faithfulness": FaithfulnessCorrelation(
        nr_runs=10,
        subset_size=10,
        perturb_baseline="black",
        perturb_func=baseline_replacement_by_indices,
        similarity_func=similarity_func.correlation_pearson,
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Complexity": Sparseness(
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Effective Complexity": EffectiveComplexity(
        return_aggregate=True,
        disable_warnings=True,
    ),
}

import time
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

    if name in ["Faithfulness", "SensitivityN"]:
        kwargs["s_batch"] = s_batch

    if name == "Robustness":
        kwargs["explain_func"] = explain_func
        kwargs["s_batch"] = s_batch

    score = metric(**kwargs)
    end_time = time.perf_counter()
    print(f"{name} took {end_time - start_time:.2f} seconds.")
    results[name] = score

# === Results Output ===
print("\n--- Evaluation Results (LIME) ---")
for name, score in results.items():
    print(f"{name}: {score}")

    