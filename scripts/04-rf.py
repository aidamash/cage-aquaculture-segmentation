import os
import sys
import glob
import joblib
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import rasterio
from skimage.io import imsave, imread
import PIL
import segmentation_models_pytorch as smp
import torch

from utils.load_images import get_images, create_mask
from utils.sample_annotations import get_annotations
from utils.metrics import compute_metrics


def get_patches(img, points, patch_size=10):
    img_h, img_w = img.shape[:2]
    X, Y = [], []
    for x, y, label in points:
        if (patch_size <= y < img_h - patch_size) and (patch_size <= x < img_w - patch_size):
            patch = img[y - patch_size:y + patch_size, x - patch_size:x + patch_size, :].reshape(1, -1)
            X.append(patch)
            Y.append(1 if label == 'pos' else 0)

    return X, Y


def get_splits(data_path, patch_size=10):
    train_points, valid_points, test_points = get_annotations(data_path, num_samples_per_polygon=2000)
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = [], [], [], [], [], []
    for img_folder in range(1, 11):
        if img_folder == 2 or img_folder == 6:
            continue

        # load image
        img = get_images(data_path, image_folder=img_folder)[0].astype(np.float32)
        for split, points, X, Y in zip(
                ['train', 'valid', 'test'], [train_points, valid_points, test_points],
                [train_X, valid_X, test_X], [train_Y, valid_Y, test_Y]
        ):
            _X, _Y = get_patches(img, points[img_folder], patch_size=patch_size)
            if len(_X) == 0:
                continue

            X.extend(np.vstack(_X))
            Y.extend(np.array(_Y))

    return np.vstack(train_X), np.array(train_Y), np.vstack(valid_X), np.array(valid_Y), np.vstack(test_X), np.array(test_Y)


def train(data_path, results_dir):
    seed = 42
    patch_size = 10
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = get_splits(data_path, patch_size=patch_size)
    print(f'num samples: train: {train_X.shape[0]} - valid: {valid_X.shape[0]} - test: {test_X.shape[0]}')

    # model
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    model.fit(train_X, train_Y)
    print(f'RF train accuracy: {model.score(train_X, train_Y)} - test accuracy: {model.score(test_X, test_Y)}')

    # save
    out_dir = get_checkpoint_dir(results_dir, save=True)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    joblib.dump(model, os.path.join(out_dir, 'rf.joblib'))


def get_checkpoint_dir(results_dir, version=None, save=True):
    if version is None:
        rf_result_dirs = sorted(glob.glob(os.path.join(results_dir, 'rf_*')))
        version = 0
        if len(rf_result_dirs) > 0:
            version = int(rf_result_dirs[-1].split('_')[-1])
            if save:
                version += 1

    return os.path.join(results_dir, f'rf_{version}')


def load_model(results_dir, version):
    out_dir = get_checkpoint_dir(results_dir, version, save=False)
    return joblib.load(os.path.join(out_dir, 'rf.joblib'))


def visualise(data_path, results_dir, version):
    out_dir = get_checkpoint_dir(results_dir, version, save=False)
    model = load_model(results_dir, version)

    # visualise
    filenames = glob.glob(os.path.join(data_path, 'dataset/test', '*.joblib'))
    patch_size = 10
    stride = 2
    metrics = defaultdict(list)
    for i, filename in enumerate(tqdm(filenames, desc='visualisation')):
        # load patch
        patch, mask = joblib.load(filename)

        points = []
        for x in range(patch_size, patch.shape[0] - patch_size - 1, stride):
            for y in range(patch_size, patch.shape[1] - patch_size - 1, stride):
                points.append((x, y, None))

        x_pred, _ = get_patches(patch, points, patch_size=patch_size)
        prob = model.predict_proba(np.vstack(x_pred))
        pr_mask = np.zeros_like(mask)
        for (x, y, _), p in zip(points, prob[:, 1]):
            if p > 0.5:
                pr_mask[y, x] = 255
                for j in range(1, stride):
                    pr_mask[y:y + stride, x + j] = 255
                    pr_mask[y + j, x:x + stride] = 255

        # measure metrics
        tp, fp, fn, tn = smp.metrics.get_stats(torch.LongTensor(pr_mask), torch.LongTensor(mask), mode='binary')
        metrics['tp'].append(tp)
        metrics['fp'].append(fp)
        metrics['fn'].append(fn)
        metrics['tn'].append(tn)

        # plot
        plt.clf()
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(patch[:, :, :3])
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask.squeeze())
        plt.title('Ground truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.squeeze())
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(out_dir, f'patch-test-{i}.png'))
        plt.close()

    # report metrics
    tp = torch.cat(metrics['tp'])
    fp = torch.cat(metrics['fp'])
    fn = torch.cat(metrics['fn'])
    tn = torch.cat(metrics['tn'])

    per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')
    dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')

    print(f'per image iou: {per_image_iou} - dataset iou: {dataset_iou}')


def predict_mask(data_path, results_dir, version):
    out_dir = get_checkpoint_dir(results_dir, version, save=False)
    model = load_model(results_dir, version)
    patch_size = 10
    patch_wh = 320
    overlap = 0.2
    for img_folder in range(1, 11):
        if img_folder == 2 or img_folder == 6:
            continue

        # load image and mask
        img, mask, annot, img_transform, img_crs = create_mask(data_path, img_folder)
        img_h, img_w = img.shape[:2]

        # create mask
        pred_mask = np.zeros((img_h, img_w), dtype=np.float32)

        i = patch_wh
        while i < img_h:
            i_e = i + patch_wh
            if i_e > img_h:
                break

            j = patch_wh
            while j < img_w:
                j_e = j + patch_wh
                if j_e > img_w:
                    break

                points = []
                for x in range(j, j_e):
                    for y in range(i, i_e):
                        points.append((x, y, None))

                x_pred, _ = get_patches(img, points, patch_size=patch_size)
                prob = model.predict_proba(np.vstack(x_pred))
                for (x, y, _), p in zip(points, prob[:, 1]):
                    pred_mask[y, x] = p

                j += int((1.0 - overlap) * patch_wh)

            i += int((1.0 - overlap) * patch_wh)

        # convert to binary
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # save
        with rasterio.open(
                os.path.join(out_dir, f'{img_folder}-pred-mask.tif'), mode='w',
                driver='GTiff', height=pred_mask.shape[0], width=pred_mask.shape[1],
                count=1, dtype=pred_mask.dtype, crs='EPSG:32736', transform=img_transform,
        ) as out_f:
            out_f.write(pred_mask, 1)

        imsave(os.path.join(out_dir, f'{img_folder}-mask.png'), 255 * mask)
        imsave(os.path.join(out_dir, f'{img_folder}-pred-mask.png'), 255 * pred_mask)
        img[:, :, 1] *= (1 + 3 * pred_mask)
        imsave(os.path.join(out_dir, f'{img_folder}-img-pred-mask.png'), np.clip(255 * img[:, :, :3], 0, 255).astype(np.uint8))


def metrics(data_path, results_dir, version):
    PIL.Image.MAX_IMAGE_PIXELS = 370000000

    out_dir = get_checkpoint_dir(results_dir, version, save=False)
    m_overlaps, p_overlaps, iou_score = defaultdict(list), defaultdict(list), defaultdict(list)
    for c_i, img_folder in enumerate(range(1, 11)):
        if img_folder == 2 or img_folder == 6:
            continue

        mask = imread(os.path.join(out_dir, f'{img_folder}-mask.png'))
        pred = imread(os.path.join(out_dir, f'{img_folder}-pred-mask.png'))

        _m_overlaps, _p_overlaps, _iou_score = compute_metrics(mask, pred, c_i, min_area_threshold=100)
        m_overlaps.update(_m_overlaps)
        p_overlaps.update(_p_overlaps)
        iou_score.update(_iou_score)

    # compute metrics
    num_m, num_p = len(m_overlaps), len(p_overlaps)
    tp = sum(len(v) > 1 for v in m_overlaps.values())
    fp = sum(len(v) == 1 for v in p_overlaps.values())
    fn = num_m - tp

    print(f'num positives: {num_m} - num predictions: {num_p}')
    print(f'tp: {tp} - fn: {fn} - fp: {fp}')
    print(f'average iou score: {np.mean(list(iou_score.values()))}')

    # compute total areas
    print(f'total ground truth area (sq-meters): {9 * sum(v[0] for v in m_overlaps.values())}')
    print(f'total prediction area (sq-meters): {9 * sum(v[0] for v in p_overlaps.values())}')


if __name__ == '__main__':
    data_path = os.path.join(os.environ['HOME'], 'projects/aida/cage-aquaculture-mapping/planet-data')
    results_dir = os.path.join(os.environ['HOME'], 'projects/aida/cage-aquaculture-mapping/results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if len(sys.argv) == 1:
        mode, version = 'train', None
    elif len(sys.argv) == 2:
        mode, version = sys.argv[1], None
    elif len(sys.argv) == 3:
        mode, version = sys.argv[1], sys.argv[2]

    if mode == 'train':
        train(data_path, results_dir)
    elif mode == 'visualise':
        visualise(data_path, results_dir, version)
    elif mode == 'predict':
        predict_mask(data_path, results_dir, version)
    elif mode == 'metrics':
        metrics(data_path, results_dir, version)
