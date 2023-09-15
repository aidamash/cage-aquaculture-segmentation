import os
import numpy as np
import matplotlib.pyplot as plt

from utils.load_images import get_images

data_path = os.path.join(os.environ['HOME'], 'projects/aida/cage-aquaculture-mapping/planet-data')

for folder in range(1, 11):
    img = get_images(data_path, image_folder=folder, brightness_ratio=6)[0]
    img_h, img_w = img.shape[:2]
    print(img_h, img_w)

    patch_wh = 1000
    num_splits_h, num_splits_w = int(np.ceil(img_h / patch_wh)), int(np.ceil(img_w / patch_wh))
    for i in range(num_splits_h):
        y = i * patch_wh
        for j in range(num_splits_w):
            x = j * patch_wh
            patch = img[y:min(y + patch_wh, img_h), x:min(x + patch_wh, img_w), :3]
            filename = os.path.join(data_path, 'annotations', f'patch-{folder}-{x}-{y}.png')
            print(f'saving: {filename}')
            plt.imsave(filename, patch)
