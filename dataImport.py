import os
import re
import sys

import imageio
import glob
from PIL import Image
import ImageUtils
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def loadPictures(folder: str, grayscale: bool = True):
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

def readTrackingBox(filepath):
    pass


def readClipInfo(filepath) -> str:
    f = open(filepath+"/clip_info.csv", 'r')
    lines = f.readlines()
    f.close()
    print(lines)
    startTime = lines[8].split(',')[1]
    endTime = lines[9].split(',')[1]
    event = lines[10].split(',')[1]
    return event

# Decapricated
def readCSVfolder(folder:str):
    print("Warning, not maintained",file=sys.stderr)
    events = []
    print("Folder", folder)
    for superDirs in glob.glob(folder+"*"):
        for clipPath in glob.glob(superDirs+"*"):
            print("ClipPath", clipPath)
            events.append(readClipInfo(clipPath+"/clip_info.csv"))
    return events

def readDataset(folderpath:str):
    with open(folderpath+"/events.pkl") as f:
        rawEvents = f.readlines()
    currentEvent:str = ""
    currentClip:int = 0

    #Expressions to check for
    clipNumberExp = re.compile("^p\d*")
    clipNameExp = re.compile("^aS.*")
    clipEventExp = re.compile("^asS.*")

    sizeOfDataset  = 572
    # Results:
    events = np.empty(sizeOfDataset,dtype=object)
    clips = np.zeros((sizeOfDataset, 20, 360, 490))


    for line in rawEvents:
        if clipEventExp.match(line):
            currentEvent = line.split("'")[1]
        elif clipNameExp.match(line):
            #print("CurrentClip",currentClip)
            #print(events)
            events[currentClip] =  readClipInfo(folderpath+"/"+line.split("'")[1]+'/')
            clips[currentClip] = loadPictures(folderpath+"/"+line.split("'")[1]+"/")
        elif clipNumberExp.match(line):
            currentClip = int(line[1:])
        else:
            print("Something is strange width: "+line,file=sys.stderr)

    return events, clips


if __name__ == '__main__':
    #print(readCSVfolder("/home/henrik/Cogito/Hackathon/data/"))

    #print(readTrackingBox("/home/henrik/Cogito/Hackathon/data/etgt5N2CSD8/clip_27/12_info.csv"))

    #print(readClipInfo("/home/henrik/Cogito/Hackathon/data/_6MvD7aK_bI/clip_18/clip_info.csv"))

    #loadPictures("/home/henrik/Cogito/Hackathon/data/etgt5N2CSD8/clip_27/*.png", True)

    readDataset("/home/henrik/Cogito/Hackathon/data")
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