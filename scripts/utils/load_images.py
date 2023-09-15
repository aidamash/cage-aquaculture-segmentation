import os
import json
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from skimage.draw import polygon


def normalize(band, ratio=4):
    min_b, max_b = np.min(band), np.max(band)
    return np.clip(ratio * (band - min_b) / (max_b - min_b), 0, 1)


def get_raw_img(data_path, image_folder):
    img_name = os.path.join(data_path, f'{image_folder}/composite.tif')

    # load image
    print(f'Loading {img_name}')
    return rasterio.open(img_name)


def get_images(data_path, image_folder='TOL', brightness_ratio=4):
    raw_img = get_raw_img(data_path, image_folder)
    bands = [normalize(raw_img.read(i), ratio=brightness_ratio) for i in raw_img.indexes]
    num_bands = len(bands)
    if num_bands == 4:
        rgb_index = (3, 2, 1)
    elif num_bands == 8:
        rgb_index = (6, 4, 2)

    img = np.dstack(
        [bands[i - 1] for i in rgb_index] +
        [bands[i - 1] for i in raw_img.indexes if i not in rgb_index] +
        [normalize((bands[rgb_index[1] - 1] - bands[-1]) / (bands[rgb_index[1] - 1] + bands[-1] + 1e-6))]  # NDWI
    )

    return img, raw_img.transform, raw_img.crs


def create_mask(data_path, img_folder, annot_round=None):
    # load annotations
    if annot_round is None:
        annot_filename = os.path.join(data_path, f'annotations/{img_folder}-polygons.json')
    else:
        annot_filename = os.path.join(data_path, f'annotations/{img_folder}-{annot_round}-polygons.json')

    with open(annot_filename, 'r') as in_f:
        annot = json.load(in_f)

    # load image
    img, img_transform, img_crs = get_images(data_path, image_folder=img_folder)
    img = img.astype(np.float32)
    img_h, img_w = img.shape[:2]

    # create mask
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for filename in annot:
        for pg in annot[filename]:
            # fill in the mask
            rr, cc = polygon(pg[1], pg[0])
            mask[rr, cc] = 255

    return img, mask, annot, img_transform, img_crs


def plot_patch_mask(patch, mask):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(patch[:, :, :3])
    plt.title('Patch')
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title('Mask')
    plt.draw()
    plt.show()
