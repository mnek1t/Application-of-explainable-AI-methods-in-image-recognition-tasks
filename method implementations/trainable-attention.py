import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
import json
from tensorflow.keras.applications.inception_v3 import InceptionV3
import quantus
import time

start_execution_time = time.perf_counter()

INCEPTION_V3_IMAGE_SIZE = (224, 224)
SPLIT_DATASET_PATH = "D:\\University\\Bachelor Thesis\\garbadge_dataset\\splitted_augmented_dataset"
TRAIN_DATASET_PATH = os.path.join(SPLIT_DATASET_PATH, "train")
TEST_DATASET_PATH = os.path.join(SPLIT_DATASET_PATH, "test")
VAL_DATASET_PATH = os.path.join(SPLIT_DATASET_PATH, "val")

base_model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs):
        attn_map = self.conv(inputs)
        attn_output = inputs * attn_map 
        return attn_output

# building classification head
x = base_model.output
x = Attention()(x)  # Apply attention
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(6, activation='softmax')(x)  # 6 classes - as in the dataset

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DATASET_PATH, target_size=INCEPTION_V3_IMAGE_SIZE, batch_size=32, class_mode='categorical')

val_generator = datagen.flow_from_directory(
    VAL_DATASET_PATH, target_size=INCEPTION_V3_IMAGE_SIZE, batch_size=32, class_mode='categorical')

test_generator = datagen.flow_from_directory(
    TEST_DATASET_PATH, target_size=INCEPTION_V3_IMAGE_SIZE, batch_size=32, class_mode='categorical', shuffle=False)

# set best model checkpoints and early stopping to prevent overtraining
checkpoint_path = "best_trainable_attention_spatial.keras"
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

best_model.save("best_trainable_attention_spatial.keras")

stopped_epoch = len(history.history['loss'])

# save history of model training to .json file
trimmed_history = {key: value[:stopped_epoch] for key, value in history.history.items()}
with open("trainable_attention_spatial_inception_history_trimmed.json", "w") as f:
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

CLASS_NAMES = ['glass', 'cardboard', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\best_trainable_attention_spatial.keras"
IMAGE_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\bcaa-trash.jpg"

def get_attention_weights(model, img):
    img = img[None, ...]  
    img = img / 255.0   

    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('attention').input  
    )
    features = feature_extractor(img)  
    attention_layer = model.get_layer("attention")  
    attention_maps = attention_layer(features, training=False)
    return attention_maps

def plot_results(title, input, img):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    attention_plot = ax[0].imshow(input, cmap='viridis') 
    ax[0].axis("off")
    ax[0].set_title(title)

    cbar = fig.colorbar(attention_plot, ax=ax[0], fraction=0.046, pad=0.04)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=15)

    ax[1].imshow(img)
    ax[1].axis("off")
    ax[1].set_title("Original Image")

    plt.tight_layout()
    plt.show()

img = image.load_img(IMAGE_PATH, target_size=(224, 224))
loaded_img = img

pixels = np.asarray(img).astype('float32')

x_batch = np.expand_dims(pixels, axis=0) 
img = image.img_to_array(img)

s_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
s_batch[:, 56:168, 56:168, :] = 1

x_batch_localisation = np.transpose(x_batch, (0, 3, 1, 2)) 
single_channel_mask = s_batch[..., 0]
s_batch_localisation = np.expand_dims(single_channel_mask, axis=1)

preds = best_model.predict(x_batch)
pred_class = np.argmax(preds[0])
y_batch = np.array([pred_class])

attention_weights = get_attention_weights(best_model, img)

attention_map = tf.reduce_mean(attention_weights, axis=-1)
attention_map = tf.squeeze(attention_map) 

attention_map_resized = tf.image.resize(attention_map[..., tf.newaxis], (224, 224))
attention_map_resized = tf.squeeze(attention_map_resized)
a_batch = attention_map_resized.numpy()

plot_results('Trainable attention map', a_batch, loaded_img)

a_batch = np.expand_dims(a_batch, axis=0)

metrics = {
    "Faithfulness": quantus.FaithfulnessCorrelation(
        nr_runs=15,
        subset_size=20,
        perturb_baseline="black",
        perturb_func= quantus.baseline_replacement_by_indices,
        similarity_func= quantus.similarity_func.correlation_pearson,
        return_aggregate=True,
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
    print(f"\nEvaluating {name} for Trainable Attention...")
    start_time = time.perf_counter()
    kwargs = {
        "model": best_model,
        "x_batch": x_batch,
        "y_batch": y_batch,
        "a_batch": a_batch
    }
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
print(f"\nTotal SHAP + Evaluation time: {end_execution_time - start_execution_time:.2f} seconds.")