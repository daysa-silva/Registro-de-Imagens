import cv2, logging
from main import stitch
from os import listdir
from os.path import isfile, join

path = '../Faculty_of_arts_and_design_I'
output_dir = '../Output'

files = list(
    filter(
        lambda x: x.find('png') > -1,
        sorted([f for f in listdir(path) if isfile(join(path,f))],reverse=True)
    )
)

queryImg_path = ''
it = 0

for file in files:
    trainImg_path = join(path, file)

    if queryImg_path == '':
        queryImg_path = trainImg_path
    else:
        queryImg = stitch(queryImg_path, trainImg_path, 'brisk', 'knn')
        queryImg_path = "%s/%d.JPG" % (output_dir, it)

        cv2.imwrite(queryImg_path, queryImg)

    it += 1