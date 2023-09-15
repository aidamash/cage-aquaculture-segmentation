import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.load_images import get_images


data_path = os.path.join(os.environ['HOME'], 'projects/aida/cage-aquaculture-mapping/planet-data')

patch_wh = 50
for img_folder in range(1, 11):
    if img_folder == 2 or img_folder == 6:
        continue

    # load annotations
    annot_filename = os.path.join(data_path, f'annotations/{img_folder}-polygons.json')
    with open(annot_filename, 'r') as in_f:
        annot = json.load(in_f)

    img = get_images(data_path, image_folder=img_folder, brightness_ratio=2)[0]
    img_h, img_w = img.shape[:2]

    for filename in annot:
        for pg in annot[filename]:
            # sample near
            x, y = np.round(pg[0]).astype(int), np.round(pg[1]).astype(int)

            min_x_pg, min_y_pg = np.min(x), np.min(y)
            max_x_pg, max_y_pg = np.max(x), np.max(y)

            min_x_p0, min_y_p0 = max(0, min_x_pg - patch_wh), max(0, min_y_pg - patch_wh)
            max_x_p0, max_y_p0 = min(img_w, max_x_pg + patch_wh), min(img_h, max_y_pg + patch_wh)

            min_x_p1, min_y_p1 = max(0, min_x_pg - 10), max(0, min_y_pg - 10)
            max_x_p1, max_y_p1 = min(img_w, max_x_pg + 10), min(img_h, max_y_pg + 10)

            patch = np.copy(img[min_y_p0:max_y_p0, min_x_p0:max_x_p0, :3])
            patch[min_y_p1 - min_y_p0:max_y_p1 - max_y_p0, min_x_p1 - min_x_p0:max_x_p1 - max_x_p0, :] *= 3

            patch_filename = os.path.join(data_path, 'annotations', f'patch-{img_folder}-{min_x_p0}-{min_y_p0}.png')
            print(f'writing {patch_filename}')
            plt.imsave(patch_filename, np.clip(patch, 0, 1))
