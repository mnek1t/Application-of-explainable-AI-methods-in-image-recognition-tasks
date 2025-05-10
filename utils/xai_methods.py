import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_path, target_size=target_size)
    pixels = image.img_to_array(img) / 255.0
    return np.expand_dims(pixels, axis=0), img

def preprocess_localisation_from_contours(image_path, target_size):
    import cv2
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s_batch = np.zeros((1, target_size[0], target_size[1], 3), dtype=np.float32)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        s_batch[:, y:y+h, x:x+w, :] = 1

    x_batch = np.transpose(np.expand_dims(img_resized / 255.0, axis=0), (0, 3, 1, 2))
    s_batch_localisation = np.expand_dims(s_batch[..., 0], axis=1)

    return x_batch.astype(np.float32), s_batch_localisation.astype(np.float32)

def plot_training_history(history_dict):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict['accuracy'], label='Train Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.show()

def plot_results(title, explanation, img, cmap = None, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(explanation, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0].axis("off")
    ax[0].set_title(title)

    ax[1].imshow(img)
    ax[1].axis("off")
    ax[1].set_title("Original Image")
    plt.tight_layout()
    plt.show()