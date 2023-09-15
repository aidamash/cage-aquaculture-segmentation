import os
import json
from collections import defaultdict
import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt

from utils.load_images import get_images

data_path = os.path.join(os.environ['HOME'], 'projects/aida/cage-aquaculture-mapping/planet-data')

# polygon annotations
for img_folder in range(1, 11):
    if img_folder == 2 or img_folder == 6:
        continue

    input_annot = os.path.join(data_path, f'annotations/{img_folder}-1.json')
    output_annot = os.path.join(data_path, f'annotations/{img_folder}-1-polygons.json')

    if not os.path.exists(input_annot):
        continue

    # convert the annotation
    with open(input_annot, 'r') as in_f:
        with open(output_annot, 'w') as out_f:
            annot = json.load(in_f)
            polygons = defaultdict(list)
            for filename in annot:
                x_offset, y_offset = os.path.splitext(filename)[0].split('-')[-2:]
                x_offset, y_offset = int(x_offset), int(y_offset)
                for region in annot[filename]['regions']:
                    x = annot[filename]['regions'][region]['shape_attributes']['all_points_x']
                    y = annot[filename]['regions'][region]['shape_attributes']['all_points_y']

                    x = [x_offset + _x for _x in x]
                    y = [y_offset + _y for _y in y]

                    label = annot[filename]['regions'][region]['region_attributes']['label']
                    polygons[filename].append([x, y, label])

            json.dump(polygons, out_f)

            # show polygons
            # img = get_images(data_path, image_folder=folder, brightness_ratio=6)[0]
            # for filename in polygons:
            #     for pg in polygons[filename]:
            #         # fill in the mask
            #         rr, cc = polygon(pg[1], pg[0])
            #         img[rr, cc, 0] *= 4

            # img = np.clip(img, 0, 1)
            # plt.imsave(os.path.join(data_path, f'{folder}-labels.png'), img[:, :, :3])
