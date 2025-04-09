import os
import splitfolders

DATASET_PATH = "D:\\University\\Bachelor Thesis\\garbadge_dataset\\Augmented_Image_Dataset" 
OUTPUT_SPLIT_PATH = "D:\\University\\Bachelor Thesis\\garbadge_dataset\\splitted_augmented_dataset"
for subdir, dirs, files in os.walk(DATASET_PATH):
    for dir in dirs:
        splitfolders.ratio(DATASET_PATH, output=OUTPUT_SPLIT_PATH, seed=1337, ratio=(.8, .1, .1), group_prefix=None)