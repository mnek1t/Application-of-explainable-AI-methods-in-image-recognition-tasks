from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

SPLIT_DATASET_PATH = "D:\\University\\Bachelor Thesis\\garbadge_dataset\\splitted_augmented_dataset"
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\results\\best_inception_model.keras"
INCEPTION_V3_IMAGE_SIZE = (224, 224)

test_dataset_path = os.path.join(SPLIT_DATASET_PATH, "test")

model = load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    test_dataset_path, target_size=INCEPTION_V3_IMAGE_SIZE, batch_size=32, class_mode='categorical', shuffle=False)
pred_probs = model.predict(test_generator)
y_pred = np.argmax(pred_probs, axis=1)

y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_labels))
print(confusion_matrix(y_true, y_pred))