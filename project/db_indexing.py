#export PYTHONPATH="/usr/local/opencv-2.4.9/lib/python2.7/site-packages/"


import os
import sys
import glob
import cv2
import numpy as np
import argparse

from timeit import default_timer as timer

''' 
    Usage :
    ./db_indexing.py -d "database_name"
    
    Example :
    ./db_indexing.py -d "base1"
'''

######## Program parameters

parser = argparse.ArgumentParser()

## Database name
parser.add_argument("-d", "--database", dest="db_name",
                    help="input image database", metavar="STRING", default="None")

args = parser.parse_args()

## Set paths
img_dir="/share/esir3/VO/Images/" + args.db_name + "/"
imagesNameList = glob.glob(img_dir+"*.jpg")
output_dir="./databases/" + args.db_name

if not os.path.exists(img_dir):
    print "The directory containing images: "+img_dir+" is not found -- EXIT\n"
    sys.exit(1)


dbDesc = []
imgIndex = []
for imgName in imagesNameList[:10] :
    img = cv2.imread(imgName)


    sift = cv2.xfeatures2d.SIFT_create()

    kp,des = sift.detectAndCompute(img,None)
    for desc in des:
        dbDesc.append(desc)
        imgIndex.append(imgName)
        
np.save(output_dir+"_dbDesc.npy",dbDesc)
np.save(output_dir+"_imageIndex",imgName)

FLANN_INDEX_ALGO=0

index_params = dict(algorithm = FLANN_INDEX_ALGO)   # for linear search

fl = cv2.flann_Index(np.asarray(dbDesc,np.float32),index_params)
fl.save(output_dir+"_linearIndex.dat")

index_params_tree = dict(algorithm = FLANN_INDEX_ALGO, trees = 5) # for kdtree search

fltree = cv2.flann_Index(np.asarray(dbDesc,np.float32),index_params_tree)
fltree.save(output_dir+"_kdtreeIndex.dat")






