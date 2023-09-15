import os
from collections import defaultdict
import json
import numpy as np
import joblib
from tqdm import tqdm

from utils.load_images import create_mask


data_path = os.path.join(os.environ['HOME'], 'projects/aida/cage-aquaculture-mapping/planet-data')

# create output directory
out_path = os.path.join(data_path, 'dataset')
if not os.path.exists(out_path):
    os.mkdir(out_path)

# settings
seed = 42
np.random.seed(seed)

train_ratio, valid_ratio = 0.7, 0.1
num_near_samples_per_polygon = 20
num_far_samples_per_polygon = 20
num_rand_samples_per_image = 50
patch_w, patch_h = 320, 320

# create patches
out_split_filenames = defaultdict(list)
for img_folder in range(1, 11):
    if img_folder == 2 or img_folder == 6:
        continue

    img, mask, annot = create_mask(data_path, img_folder)[:3]
    img_h, img_w = img.shape[:2]

    # split patches for train, validation, and test
    filenames = list(annot.keys())
    np.random.shuffle(filenames)
    num_files = len(filenames)
    num_train, num_val = int(train_ratio * num_files), int(max(1, valid_ratio * num_files))
    train_files, valid_files, test_files = filenames[:num_train], filenames[num_train:num_train + num_val], filenames[num_train + num_val:]
    print(f'Num train files: {len(train_files)}, valid files: {len(valid_files)}, test: {len(test_files)}')

    out_split_filenames['train'].extend(train_files)
    out_split_filenames['valid'].extend(valid_files)
    out_split_filenames['test'].extend(test_files)

    # create splits
    for split, split_files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        # create split directory
        split_path = os.path.join(out_path, split)
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        for filename in split_files:
            for i, pg in enumerate(annot[filename]):
                # sample near
                x, y = np.round(pg[0]).astype(int), np.round(pg[1]).astype(int)
                min_x, min_y, max_x, max_y = np.min(x), np.min(y), np.max(x), np.max(y)
                w, h = max_x - min_x, max_y - min_y
                dx, dy = patch_w - w, patch_h - h

                for j in tqdm(range(num_near_samples_per_polygon), desc=f'{split}-{filename}-{i}-near'):
                    rand_x = max(0, min_x - np.random.randint(0, dx))
                    rand_y = max(0, min_y - np.random.randint(0, dy))

                    patch = img[rand_y:rand_y + patch_h, rand_x:rand_x + patch_w, :]
                    patch_mask = mask[rand_y:rand_y + patch_h, rand_x:rand_x + patch_w]

                    if patch.shape[0] != patch_h or patch.shape[1] != patch_w:
                        continue

                    out_filename = os.path.join(split_path, f'{filename}-near-{i}-{j}.joblib')
                    joblib.dump((patch, patch_mask), out_filename)

                # sample far
                for j in tqdm(range(num_near_samples_per_polygon), desc=f'{split}-{filename}-{i}-far'):
                    sign = 1 if np.random.randn() > 0.0 else -1
                    rand_x = max(0, min_x + sign * np.random.randint(dx, 2 * dx))
                    sign = 1 if np.random.randn() > 0.0 else -1
                    rand_y = max(0, min_y + sign * np.random.randint(dy, 2 * dy))

                    patch = img[rand_y:rand_y + patch_h, rand_x:rand_x + patch_w, :]
                    patch_mask = mask[rand_y:rand_y + patch_h, rand_x:rand_x + patch_w]

                    if patch.shape[0] != patch_h or patch.shape[1] != patch_w:
                        continue

                    out_filename = os.path.join(split_path, f'{img_folder}-{filename}-far-{i}-{j}.joblib')
                    joblib.dump((patch, patch_mask), out_filename)

        # sample random
        for i in tqdm(range(num_rand_samples_per_image), desc=f'{split}-rand'):
            rand_x = np.random.randint(patch_w, img_w - 2 * patch_w)
            rand_y = np.random.randint(patch_h, img_h - 2 * patch_h)

            patch = img[rand_y:rand_y + patch_h, rand_x:rand_x + patch_w, :]
            patch_mask = mask[rand_y:rand_y + patch_h, rand_x:rand_x + patch_w]

            if patch.shape[0] != patch_h or patch.shape[1] != patch_w:
                continue

            out_filename = os.path.join(split_path, f'{img_folder}-rand-{i}.joblib')
            joblib.dump((patch, patch_mask), out_filename)


with open(os.path.join(out_path, 'splits.json'), 'w') as out_f:
    json.dump(out_split_filenames, out_f)
