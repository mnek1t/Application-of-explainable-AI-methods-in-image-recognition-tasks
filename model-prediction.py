import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils.xai_constants import MODEL_PATH, IMAGE_PATH, TARGET_SIZE, CLASS_NAMES

model = load_model(MODEL_PATH)

img = image.load_img(IMAGE_PATH, target_size=TARGET_SIZE)
pixels = image.img_to_array(img) / 255.0
x_batch = np.expand_dims(pixels, axis=0)

preds = model.predict(x_batch)
pred_class = np.argmax(preds[0])

print("\n--- Model Prediction ---")
print(f"Predicted class index: {pred_class}")
print(f"Predicted class label: {CLASS_NAMES[pred_class]}")
print(f"Prediction confidence scores:")
for idx, class_name in enumerate(CLASS_NAMES):
    print(f"\t{class_name}: {preds[0][idx]:.4f}")