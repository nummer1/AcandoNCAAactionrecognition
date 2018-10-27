import imageio
import glob
from PIL import Image
import ImageUtils
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def loadPictures(, folder: str, grayscale: bool = True):
    pictures = []
    grey = []
    for im_path in glob.glob(folder):
        im = imageio.imread(im_path)
        print(im.shape)
        pictures.append(im)

    pictureWidth = pictures[0].shape[0]
    pictureHeight = pictures[0].shape[1]

    print(pictureWidth,", ",pictureHeight)


    grey = np.zeros((len(pictures),pictureWidth,pictureHeight))
    for i in range(len(pictures)):
        for x in range(pictureWidth):
            for y in range(pictureHeight):
                #print(pictures[x][y])
                #print("Color: ",pictures[i][x][y], " , Grey: ", rgb2gray(pictures[i][x][y]))
                grey[i][x,y] = rgb2gray(pictures[i][x, y])

    if grayscale:
        return grey
    else:
        return np.array(pictures)

def readCSV(filepath):
    #for csv_path in glob.glob("/home/henrik/Cogito/Hackathon/data/etgt5N2CSD8/clip_27/*_info.csv"):
    f = open(filepath, 'r')
    lines = f.readlines()

    startTime = lines[8][1]
    endTime = lines[9][1]
    event = lines[10][1]
    return event

def readCSVfolder(folder:str):
    pass

print(readCSV("/home/henrik/Cogito/Hackathon/data/etgt5N2CSD8/clip_27/12_info.csv"))
loadPictures("/home/henrik/Cogito/Hackathon/data/etgt5N2CSD8/clip_27/*.png", True)
#
# print("------")
# # X, labels = ImageUtils.read_images("/home/henrik/Cogito/Hackathon/data/etgt5N2CSD8/clip_27/*.png")
# # X = np.array([np.array(image) for image in X])
# # X = ImageUtils.numapy_grayscale(X)
#
#
# #X, labels = ImageUtils.read_images("data")
# #X = np.array([np.array(image) for image in X])
# #X = ImageUtils.numapy_grayscale(X)
#
# #print(pictures[0])
# print(grey)/