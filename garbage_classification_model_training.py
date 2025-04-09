from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import json

# locations of dataset folders
splitted_dataset_path = "D:\\University\\Bachelor Thesis\\garbadge_dataset\\splitted_augmented_dataset"
train_dataset_path = os.path.join(splitted_dataset_path, "train")
test_dataset_path = os.path.join(splitted_dataset_path, "test")
val_dataset_path = os.path.join(splitted_dataset_path, "val")

base_model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
INCEPTION_V3_IMAGE_SIZE = (224, 224)

# freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# building classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(6, activation='softmax')(x)  # 6 classes - as in the dataset

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dataset_path, target_size=INCEPTION_V3_IMAGE_SIZE, batch_size=32, class_mode='categorical')

val_generator = datagen.flow_from_directory(
    val_dataset_path, target_size=INCEPTION_V3_IMAGE_SIZE, batch_size=32, class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_dataset_path, target_size=INCEPTION_V3_IMAGE_SIZE, batch_size=32, class_mode='categorical', shuffle=False)

# set best model checpoints and early stopping to prevent overtraining
checkpoint_path = "best_inception_model.keras"
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks
)

# saving the model
best_model = load_model(checkpoint_path)

loss, acc = best_model.evaluate(test_generator)
print(f"Best Test Accuracy (after EarlyStopping): {acc:.2f}")

best_model.save("inception_trash_classifier_best.keras")

stopped_epoch = len(history.history['loss'])

# save history of model training to .json file
trimmed_history = {key: value[:stopped_epoch] for key, value in history.history.items()}
with open("inception_history_trimmed.json", "w") as f:
    json.dump(trimmed_history, f)

# plot model training analysis
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(trimmed_history['accuracy'], label='Train Accuracy')
plt.plot(trimmed_history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(trimmed_history['loss'], label='Train Loss')
plt.plot(trimmed_history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.show()