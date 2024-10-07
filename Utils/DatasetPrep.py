#this code creates augmentations for the training data (applying same augmentations to groundtruths and labels)


import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import slidingwindow as sw
import shutil
import time
from tqdm import tqdm

def preprocess_mask(mask): #this binarizes the mask
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 255
    return mask

def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None):
    cols = 3 if predicted_masks else 2
    rows = len(images_filenames)
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
    for i, image_filename in enumerate(images_filenames):
        image = cv2.imread(os.path.join(images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(masks_directory, image_filename.replace(".tif", "_label.png")))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = preprocess_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Ground truth mask")

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

        if predicted_masks:
            predicted_mask = predicted_masks[i]
            ax[i, 2].imshow(predicted_mask, interpolation="nearest")
            ax[i, 2].set_title("Predicted mask")
            ax[i, 2].set_axis_off()
    plt.tight_layout()
    plt.show()

start = time.time()

#defining directories
dataset_directory = "E:\\CoralReefRockSponge\\TRAINING_DATA"
root_directory = os.path.join(dataset_directory)
images_directory = os.path.join(root_directory, "IMAGES")
masks_directory = os.path.join(root_directory, "LABELS")

file_list = []
for item in os.listdir(images_directory):
    if not item.startswith('.') and os.path.isfile(os.path.join(images_directory, item)):
        file_list.append(item)

images_filenames = list(sorted(file_list))
correct_images_filenames = [i for i in images_filenames if cv2.imread(os.path.join(images_directory, i)) is not None]

#display_image_grid(images_filenames, images_directory, masks_directory)

#create dir to save cropped images 'IMAGES_CROPPED'
CROP_DIR = os.path.join(root_directory, 'IMAGES_CROPPED')
CROP_IMG_DIR = os.path.join(CROP_DIR,'RAW_CROPPED')
CROP_LABEL_DIR = os.path.join(CROP_DIR,'LABELS_CROPPED')
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(CROP_IMG_DIR, exist_ok=True)
os.makedirs(CROP_LABEL_DIR, exist_ok=True)

#create dir to save cropped images augmented 'IMAGES_CROPPED_AUG'
CROP_AUG_IMG_DIR = os.path.join(CROP_DIR,'RAW_AUG_CROPPED')
CROP_AUG_LABEL_DIR = os.path.join(CROP_DIR,'LABELS_AUG_CROPPED')
os.makedirs(CROP_AUG_IMG_DIR, exist_ok=True)
os.makedirs(CROP_AUG_LABEL_DIR, exist_ok=True)



patch_size = 256
overlap = 0.05
#read raw image and corresponding label

for idx in tqdm(range(0, len(images_filenames))):

    img_name = images_filenames[idx]
    img_raw = cv2.imread(os.path.join(images_directory,img_name))

    label_name = images_filenames[idx].replace(".tif", "_label.png")
    img_label = cv2.imread(os.path.join(masks_directory, label_name))
    img_label=cv2.cvtColor(img_label,cv2.COLOR_BGR2GRAY) #import as grey 0-255
    img_label = preprocess_mask(img_label) #convert non 0 to 255 (make black and white)

    windows = sw.generate(img_raw, sw.DimOrder.HeightWidthChannel, patch_size, overlap)

    # Do stuff with the generated windows #save all into new folder of processed img after loop

    #save the cropped images and labels without augmentations
    for item in range(0,len(windows)):
        subset_raw = img_raw[windows[item].indices()]
        subset_label = img_label[windows[item].indices()]

        subset_raw_name = img_name.split('_raw.tif')[0] + '_crop_' + str(item) + '_raw.tif'
        subset_label_name = label_name.split('_label.png')[0] + '_crop_' + str(item) + '_label.png'

        #saving cropped originals
        cv2.imwrite(os.path.join(CROP_LABEL_DIR, subset_label_name), subset_label)
        cv2.imwrite(os.path.join(CROP_IMG_DIR, subset_raw_name),subset_raw)


    #now do some transformations to each image and also save cropped
    # do augmentations (flips etc)
    transforms = [
        lambda m: np.fliplr(m),
        lambda m: np.flipud(m),
        lambda m: np.rot90(m, k=1, axes=(0, 1)),
        lambda m: np.rot90(m, k=3, axes=(0, 1))
    ]

    windows_aug = sw.generate(img_raw, sw.DimOrder.HeightWidthChannel, patch_size, overlap, transforms)

    for item in range(0, len(windows_aug)):

        transformed_raw = windows_aug[item].apply(img_raw)
        transformed_raw_name = img_name.split('.tif')[0] + '_crop_aug_' + str(item) + '_raw.tif'

        transformed_label = windows_aug[item].apply(img_label)
        transformed_label_name =  label_name.split('_label.png')[0] + '_crop_aug_' + str(item) + '_label.png'

        #saving cropped augmented
        cv2.imwrite(os.path.join(CROP_AUG_LABEL_DIR, transformed_label_name), transformed_label)
        cv2.imwrite(os.path.join(CROP_AUG_IMG_DIR, transformed_raw_name), transformed_raw)

end = time.time()

print(f"time elapsed = {(end-start)/60} min")
