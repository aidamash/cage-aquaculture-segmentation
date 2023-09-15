import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd


# add samples from polygon
# ref: https://www.matecdev.com/posts/random-points-in-polygon.html
def random_points_in_bounds(min_x, min_y, max_x, max_y, num):
    x = np.random.uniform(min_x, max_x, num)
    y = np.random.uniform(min_y, max_y, num)
    return x, y


def random_points_in_polygon_bounds(polygon, num):
    min_x, min_y, max_x, max_y = polygon.bounds
    return random_points_in_bounds(min_x, min_y, max_x, max_y, num)


def sample_from_polygons(annot_polygons, num_samples_per_polygon, neg_wh_ratio=3):
    polygon_samples = []

    for x, y, label in annot_polygons:
        # sample positives
        polygon = Polygon(list(zip(x, y)))
        gdf_poly = gpd.GeoDataFrame(index=['my_poly'], geometry=[polygon])
        _x, _y = random_points_in_polygon_bounds(polygon, num_samples_per_polygon)

        df = pd.DataFrame()
        df['points'] = list(zip(_x, _y))
        df['points'] = df['points'].apply(Point)
        gdf_points = gpd.GeoDataFrame(df, geometry='points')
        sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate='within', how='left')

        # keep points in "my_poly"
        pnts_in_poly = gdf_points[sjoin.index_right == 'my_poly']
        for row in pnts_in_poly.iterrows():
            polygon_samples.append((
                int(np.round(row[1]['points'].x)),
                int(np.round(row[1]['points'].y)),
                label
            ))

        # sample negatives
        min_x, min_y, max_x, max_y = polygon.bounds
        pw, ph = max_x - min_x, max_y - min_y
        neg_pw, neg_ph = neg_wh_ratio * pw, neg_wh_ratio * ph
        min_x -= neg_pw // 2
        max_x += neg_pw // 2
        min_y -= neg_ph // 2
        max_y += neg_ph // 2

        _x, _y = random_points_in_bounds(min_x, min_y, max_x, max_y, neg_wh_ratio * num_samples_per_polygon)
        df = pd.DataFrame()
        df['points'] = list(zip(_x, _y))
        df['points'] = df['points'].apply(Point)
        gdf_points = gpd.GeoDataFrame(df, geometry='points')
        sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate='within', how='left')

        # keep points in "my_poly"
        pnts_not_in_poly = gdf_points[sjoin.index_right != 'my_poly']
        for row in pnts_not_in_poly.iterrows():
            polygon_samples.append((
                int(np.round(row[1]['points'].x)),
                int(np.round(row[1]['points'].y)),
                'neg'
            ))

    return polygon_samples


def get_annotations(data_path, num_samples_per_polygon=2000):
    seed = 42
    np.random.seed(seed)

    # load splits
    with open(os.path.join(data_path, 'dataset/splits.json'), 'r') as in_f:
        splits_files = json.load(in_f)

    # load annotations
    annot_polygons = {}
    for img_folder in range(1, 11):
        if img_folder == 6:
            continue

        annot_filename = os.path.join(data_path, f'annotations/{img_folder}-polygons.json')
        with open(annot_filename, 'r') as in_f:
            annot_polygons[img_folder] = json.load(in_f)

    # create splits
    train_points, valid_points, test_points = defaultdict(list), defaultdict(list), defaultdict(list)
    train_count, valid_count, test_count = 0, 0, 0
    for split, points in zip(['train', 'valid', 'test'], [train_points, valid_points, test_points]):
        for img_folder in range(1, 11):
            if img_folder == 6:
                continue

            # create point samples
            for filename in splits_files[split]:
                # create polygon samples
                if filename not in annot_polygons[img_folder]:
                    continue

                pnts = sample_from_polygons(annot_polygons[img_folder][filename], num_samples_per_polygon)
                points[img_folder].extend(pnts)
                if split == 'train':
                    train_count += len(pnts)
                elif split == 'valid':
                    valid_count += len(pnts)
                if split == 'test':
                    test_count += len(pnts)

    print(f'Num annotations: train = {train_count}, valid: {valid_count}, test = {test_count}')

    return train_points, valid_points, test_points


if __name__ == '__main__':
    data_path = os.path.join(os.environ['HOME'], 'projects/aida/cage-aquaculture-mapping/planet-data')
    get_annotations(data_path)
