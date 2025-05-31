import albumentations as A
import cv2
import os
import uuid

# Initialize the paths and parameters
INPUT_DATASET_PATH = 'D:\\University\\Bachelor Thesis\\garbadge_dataset\\TrashType_Image_Dataset'
OUTPUT_DATASET_PATH = 'D:\\University\\Bachelor Thesis\\garbadge_dataset\\Augmented_Image_Dataset'
TARGET_IMAGE_AMOUNT = 700

# Chain of augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),                            # 50% that image is flipped horizontally 
    A.RandomBrightnessContrast(p=0.2),                  # 20% that brightness and contrast are changed
    A.Rotate(limit=25, p=0.5),                          # 50% that image is rotated between -25 and 25 degrees
    A.RandomCrop(width=224, height=224, p=0.5),         # 50% that image is cropped to 224x224 pixels
    A.Blur(blur_limit=3, p=0.3),                        # 30% that image is blurred with a kernel size of 3
    A.ColorJitter(p=0.3),                               # 30% that color is changed
])

# Copies images from source to destination if not already present
def copy_files(image_file_names, class_input_path, class_output_path):
    for file_name in image_file_names:
        src_path = os.path.join(class_input_path, file_name)
        dst_path = os.path.join(class_output_path, file_name)
        if os.path.exists(dst_path):
            print(f"Skipped: {dst_path} already exists.")
            continue

        img = cv2.imread(src_path)
        if img is None:
            continue

        cv2.imwrite(dst_path, img)
# Applies augmentation to increase the number of images to the target amount
def appy_augmentation(image_file_names, images_to_augment, class_input_path, class_output_path):
    augmented = 0

    while augmented < images_to_augment:
        for file_name in image_file_names:
            if augmented >= images_to_augment:
                break
            src_path = os.path.join(class_input_path, file_name)
            img = cv2.imread(src_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            augmented_image = transform(image=img)['image']
            save_file_name = file_name + '_aug_' + str(uuid.uuid4()) + '.jpg' # name of augmented image is unique
            dst_path = os.path.join(class_output_path, save_file_name)
            cv2.imwrite(dst_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
            augmented += 1
# Process each class subfolder in the dataset
for class_name in os.listdir(INPUT_DATASET_PATH):
    class_input_path = os.path.join(INPUT_DATASET_PATH, class_name)
    class_output_path = os.path.join(OUTPUT_DATASET_PATH, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    image_file_names = [f for f in os.listdir(class_input_path) if f.endswith(('.jpg'))] # dataset contains only .jpg files
    original_count = len(image_file_names)

    copy_files(image_file_names, class_input_path, class_output_path)

    if original_count < TARGET_IMAGE_AMOUNT:
        images_to_augment = TARGET_IMAGE_AMOUNT - original_count
        appy_augmentation(image_file_names, images_to_augment, class_input_path, class_output_path)
    else:
        print(f"Class '{class_name}' already has {original_count} images, no augmentation needed.")