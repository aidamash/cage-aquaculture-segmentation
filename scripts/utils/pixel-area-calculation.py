import cv2
import numpy as np
import rasterio
import os

img_path = "/Users/aida/Documents/Modules/Thesis/lake-victoria/plots/"
img_name = "-pred-mask.tif"


# img_cat = ['TOL', 'SOL']
#
# for i in img_cat:
#     image = cv2.imread(img_path + f'{i}' + img_name)
#     print(image.shape)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
#
#
#     pixels = cv2.countNonZero(thresh)
#     #pixels = len(np.column_stack(np.where(thresh > 0)))
#
#     image_area = image.shape[0] * image.shape[1]
#     area_ratio = (pixels / image_area) * 100
#
#     print('pixels', pixels)
#     print('area ratio', area_ratio)
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)


"""
TOL
(14176, 8773, 3)
pixels 119122
area ratio 0.09578337650481585
SOL
(32445, 11287, 3)
pixels 122102
area ratio 0.033342370578868276

"""

img_cat = ['TOL']#, 'SOL']

for i in img_cat:
    image = cv2.imread(img_path + f'{i}' + img_name)

    print(image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    total = 0

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [c], [255,255,255])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        pixels = cv2.countNonZero(mask)
        area = pixels * 9.0
        print("pixels and area", pixels, area)
        total += pixels
        cv2.putText(image, '{}'.format(area), (x,y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    print(total)
    # cv2.imshow('thresh', thresh)
    # Saving the image
    #cv2.imshow('image', mask)
    #cv2.imwrite(img_path + f'{i}' + 'cagePredictionArea.tif', image)
    with rasterio.open(
            os.path.join(img_path, f'{i}-cagePredictionArea2.tif'), mode='w',
            driver='GTiff', height=image.shape[0], width=image.shape[1],
            count=1, dtype=image.dtype, crs='+proj=latlong', transform=image.transform,
    ) as out_f:
        out_f.write(image, 1)



