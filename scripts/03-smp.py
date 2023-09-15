import os
import sys
import glob
from collections import defaultdict
from pprint import pprint
import joblib
import numpy as np
import torch
import albumentations as A
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
import rasterio
from skimage.io import imsave, imread
import PIL
import pandas as pd

from utils.load_images import create_mask
from utils.metrics import compute_metrics


def get_augmentation_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.MultiplicativeNoise(p=0.2)
    ])


class CageDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, split='train', transforms=None):
        self.data_path = data_path

        if split not in {'train', 'valid', 'test'}:
            raise ValueError(f'Unknown split: {split}')

        self.split = split
        self.transforms = transforms

        self.filenames = glob.glob(os.path.join(data_path, 'dataset', split, '*.joblib'))
        print(f'Dataset {split}: num samples: {len(self.filenames)}')

        seed = 42
        np.random.seed(seed)
        np.random.shuffle(self.filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, ind):
        patch, mask = joblib.load(self.filenames[ind])

        patch = patch.astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        if self.transforms:
            sample = self.transforms(image=patch, mask=mask)
            patch, mask = sample['image'], sample['mask']

        patch = np.moveaxis(patch, -1, 0)
        mask = np.expand_dims(mask > 0, 0)

        return {'image': patch, 'mask': mask}


class CageModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.create_model(
            arch, encoder_name=encoder_name, encoder_weights=encoder_weights,
            in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        self.register_buffer('mean', torch.zeros((1, in_channels, 1, 1)))
        self.register_buffer('std', torch.ones((1, in_channels, 1, 1)))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.metrics = defaultdict(list)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        pred = self.model(image)
        return pred

    def shared_step(self, batch, stage):
        image, mask = batch['image'], batch['mask']

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode='binary')
        metrics = {
            'loss': loss,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }
        self.metrics[stage].append(metrics)

        return metrics

    def shared_epoch_end(self, stage):
        # aggregate step metics
        tp = torch.cat([x['tp'] for x in self.metrics[stage]])
        fp = torch.cat([x['fp'] for x in self.metrics[stage]])
        fn = torch.cat([x['fn'] for x in self.metrics[stage]])
        tn = torch.cat([x['tn'] for x in self.metrics[stage]])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with 'empty' images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')

        metrics = {
            f'{stage}_per_image_iou': per_image_iou,
            f'{stage}_dataset_iou': dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def on_train_epoch_end(self):
        return self.shared_epoch_end('train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'valid')

    def on_validation_epoch_end(self):
        return self.shared_epoch_end('valid')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def on_test_epoch_end(self):
        return self.shared_epoch_end('test')

    def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.0001)


def get_dataloaders(data_path):
    transforms = get_augmentation_transforms()

    train_dataloader = torch.utils.data.DataLoader(CageDataset(data_path, split='train', transforms=transforms), batch_size=16, num_workers=8)
    valid_dataloader = torch.utils.data.DataLoader(CageDataset(data_path, split='valid'), batch_size=4, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(CageDataset(data_path, split='test'), batch_size=4, num_workers=8)

    return train_dataloader, valid_dataloader, test_dataloader


def get_model(args):
    model_type, encoder = 'Unet', 'resnet34'
    if isinstance(args, list):
        if len(args) == 1:
            model_type = args[0]
        elif len(args) == 2:
            model_type, encoder = args[0], args[1]

    print(f'training: model type: {model_type} - encoder: {encoder}')
    return CageModel(model_type, encoder, None, in_channels=9, out_classes=1)


def train(data_path, results_dir, args):
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(data_path)
    model = get_model(args)

    # train
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        accelerator=device, max_epochs=1000, default_root_dir=results_dir
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # validate
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    pprint(test_metrics)


def get_checkpoint_path(results_dir, args):
    version = 'version_0'
    if isinstance(args, list):
        version = args[0]

    return os.path.join(results_dir, f'lightning_logs/{version}/checkpoints')


def load_model(results_dir, args, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_dir = get_checkpoint_path(results_dir, args)
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*.ckpt')))

    return CageModel.load_from_checkpoint(checkpoints[-1], map_location=torch.device(device))


def visualise(data_path, results_dir, args):
    test_dataloader = get_dataloaders(data_path)[-1]
    model = load_model(results_dir, args)
    checkpoint_dir = get_checkpoint_path(results_dir, args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        model.eval()
        c = 0
        metrics = defaultdict(list)
        for batch in tqdm(test_dataloader, desc='visualisation'):
            logits = model(batch['image'].to(device))
            pr_masks = logits.sigmoid()

            for image, gt_mask, pr_mask in zip(batch['image'], batch['mask'], pr_masks):
                # measure metrics
                tp, fp, fn, tn = smp.metrics.get_stats(pr_mask.cpu().long(), gt_mask.cpu().long(), mode='binary')
                metrics['tp'].append(tp)
                metrics['fp'].append(fp)
                metrics['fn'].append(fn)
                metrics['tn'].append(tn)

                # plot
                plt.clf()
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(image.cpu().numpy().transpose(1, 2, 0)[:, :, :3])
                plt.title('Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(gt_mask.cpu().numpy().squeeze())
                plt.title('Ground truth')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pr_mask.cpu().numpy().squeeze())
                plt.title('Prediction')
                plt.axis('off')

                plt.savefig(os.path.join(checkpoint_dir, f'patch-test-{c}.png'))
                plt.close()
                c += 1

        # report metrics
        tp = torch.cat(metrics['tp'])
        fp = torch.cat(metrics['fp'])
        fn = torch.cat(metrics['fn'])
        tn = torch.cat(metrics['tn'])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')

        print(f'per image iou: {per_image_iou} - dataset iou: {dataset_iou}')


def predict_mask(data_path, results_dir, args):
    model = load_model(results_dir, args)
    checkpoint_dir = get_checkpoint_path(results_dir, args)
    patch_wh = 4 * 320
    overlap = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        model.eval()
        for img_folder in range(1, 11):
            if img_folder == 2 or img_folder == 6:
                continue

            # load image and mask
            img, mask, annot, img_transform, img_crs = create_mask(data_path, img_folder)
            img_h, img_w = img.shape[:2]

            # create mask
            pred_mask = np.zeros((img_h, img_w), dtype=np.float32)

            i = 0
            while i < img_h:
                i_e = i + patch_wh
                if i_e > img_h:
                    break

                j = 0
                while j < img_w:
                    j_e = j + patch_wh
                    if j_e > img_w:
                        break

                    patch = img[i:i_e, j:j_e, :]
                    logits = model(torch.FloatTensor(np.moveaxis(patch, -1, 0)).to(device))
                    pr_mask = logits.sigmoid()
                    pred_mask[i:i_e, j:j_e] = pr_mask.cpu().numpy().squeeze()

                    j += int((1.0 - overlap) * patch_wh)

                i += int((1.0 - overlap) * patch_wh)

            # convert to binary
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

            # save
            with rasterio.open(
                    os.path.join(checkpoint_dir, f'{img_folder}-pred-mask.tif'), mode='w',
                    driver='GTiff', height=pred_mask.shape[0], width=pred_mask.shape[1],
                    count=1, dtype=pred_mask.dtype, crs='EPSG:32736', transform=img_transform,
            ) as out_f:
                out_f.write(pred_mask, 1)

            imsave(os.path.join(checkpoint_dir, f'{img_folder}-mask.png'), 255 * mask)
            imsave(os.path.join(checkpoint_dir, f'{img_folder}-pred-mask.png'), 255 * pred_mask)
            img[:, :, 1] *= (1 + 3 * pred_mask)
            imsave(os.path.join(checkpoint_dir, f'{img_folder}-img-pred-mask.png'), np.clip(255 * img[:, :, :3], 0, 255).astype(np.uint8))


def metrics(data_path, results_dir, version):
    PIL.Image.MAX_IMAGE_PIXELS = 370000000

    checkpoint_dir = get_checkpoint_path(results_dir, version)
    m_overlaps, p_overlaps, iou_score = defaultdict(list), defaultdict(list), []
    for c_i, img_folder in enumerate(range(1, 11)):
        if img_folder == 2 or img_folder == 6:
            continue

        mask = imread(os.path.join(checkpoint_dir, f'{img_folder}-mask.png'))
        pred = imread(os.path.join(checkpoint_dir, f'{img_folder}-pred-mask.png'))

        mask[mask > 0] = 1
        pred[pred > 0] = 1

        _m_overlaps, _p_overlaps = compute_metrics(mask, pred, c_i, min_area_threshold=100)
        m_overlaps.update(_m_overlaps)
        p_overlaps.update(_p_overlaps)

        tp = (mask * pred).sum()
        fp = pred.sum() - tp
        fn = mask.sum() - tp
        tn = mask.shape[0] * mask.shape[1] - (tp + fp + fn)

        iou_score.append(tp / (tp + fp + fn))

    # compute polygon metrics
    num_m, num_p = len(m_overlaps), len(p_overlaps)
    tp = sum(len(v) > 1 for v in m_overlaps.values())
    fp = sum(len(v) == 1 for v in p_overlaps.values())
    fn = num_m - tp

    print(f'num positives: {num_m} - num predictions: {num_p}')
    print(f'tp: {tp} - fn: {fn} - fp: {fp}')

    # compute total areas
    print(f'total ground truth area (sq-meters): {9 * sum(v[0] for v in m_overlaps.values())}')
    print(f'total prediction area (sq-meters): {9 * sum(v[0] for v in p_overlaps.values())}')

    # compute mask metrics
    print(f'average iou score: {np.mean(iou_score)}')


def plot_training_curves(results_dir, version):
    checkpoint_dir = get_checkpoint_path(results_dir, version)
    df = pd.read_csv(os.path.join(checkpoint_dir, 'metrics.csv'))

    plt.plot(df['train_per_image_iou'].dropna())
    plt.plot(df['train_dataset_iou'].dropna())
    plt.plot(df['valid_per_image_iou'].dropna())
    plt.plot(df['valid_dataset_iou'].dropna())
    plt.legend(['train_per_image_iou', 'train_dataset_iou', 'valid_per_image_iou', 'valid_dataset_iou'])
    plt.xlabel('epochs')
    plt.ylabel('scores')
    plt.draw()
    plt.savefig(os.path.join(checkpoint_dir, 'training-scores.png'))


if __name__ == '__main__':
    data_path = os.path.join(os.environ['HOME'], 'projects/aida/cage-aquaculture-mapping/planet-data')
    results_dir = os.path.join(os.environ['HOME'], 'projects/aida/cage-aquaculture-mapping/results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if len(sys.argv) == 1:
        mode, args = 'train', None
    elif len(sys.argv) == 2:
        mode, args = sys.argv[1], None
    elif len(sys.argv) >= 3:
        mode, args = sys.argv[1], sys.argv[2:]

    if mode == 'train':
        train(data_path, results_dir, args)
    elif mode == 'visualise':
        visualise(data_path, results_dir, args)
    elif mode == 'predict':
        predict_mask(data_path, results_dir, args)
    elif mode == 'metrics':
        metrics(data_path, results_dir, args)
    elif mode == 'curves':
        plot_training_curves(results_dir, args)
