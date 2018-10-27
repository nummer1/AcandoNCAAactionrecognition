import os
import re
import sys

import imageio
import glob
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def loadPictures(folder: str, grayscale: bool = True):
    pictures = []
    grey = []
    #print("Load pictures Folder:", folder)
    for im_path in glob.glob(folder+"/*.png"):
        #print("IM_path:",im_path)
        im = imageio.imread(im_path)
        #print(im.shape)
        pictures.append(im)

    pictureWidth = pictures[0].shape[0]
    pictureHeight = pictures[0].shape[1]

    #print(pictureWidth,", ",pictureHeight)


    grey = np.zeros((len(pictures),pictureWidth,pictureHeight))
    for i in range(len(pictures)):
        for x in range(pictureWidth):
            for y in range(pictureHeight):
                ##print(pictures[x][y])
                ##print("Color: ",pictures[i][x][y], " , Grey: ", rgb2gray(pictures[i][x][y]))
                grey[i][x,y] = rgb2gray(pictures[i][x, y])

    if grayscale:
        return grey
    else:
        return np.array(pictures)

def readTrackingBox(filepath):
    pass


def readClipInfo(filepath) -> (str,str,str):
    f = open(filepath+"/clip_info.csv", 'r')
    lines = f.readlines()
    f.close()
    #print(lines)
    startTime = lines[8].split(',')[1]
    endTime = lines[9].split(',')[1]
    event = lines[10].split(',')[1]
    trainValTest = lines[11].split(',')[1]
    return event, startTime, endTime, trainValTest

# TrainValTest is 0 for training se, 1 for validation set and 2 for testset. Use it to choose which dataset you want in return.
def readData(folder:str, trainValTestReturn:int = 0):
    # Results: [[train],[val],[test]]
    events = [[],[],[]]
    clips = [[],[],[]]
    startTimes = [[],[],[]]
    endTimes = [[],[],[]]

    #print("Folder\t", folder)
    for youtubeVids in glob.glob(folder+"/*"):
        #print("Youtube videos:\t",youtubeVids)
        for clipPath in glob.glob(youtubeVids+"/*"):
            #print("ClipPath\t", clipPath)

            csv_info = readClipInfo(clipPath)
            trainValTest = csv_info[3].strip()
            if trainValTest == 'train':
                trainValTestIndex = 0
            elif trainValTest == 'val':
                trainValTestIndex = 1
            elif trainValTest == 'test':
                trainValTestIndex = 2
            else:
                #print("Some error in the trainValTest from the csv:",trainValTest,file=sys.stderr)

            events[trainValTestIndex].append(csv_info[0])
            startTimes[trainValTestIndex].append(csv_info[1])
            endTimes[trainValTestIndex].append(csv_info[2])

            clips.append(loadPictures(clipPath+"/"))

    return clips[trainValTestReturn], events[trainValTestReturn], startTimes[trainValTestReturn], endTimes[trainValTestReturn]

def readDataset(folderpath:str):
    #print("Warning: Decaprecated", file=sys.stderr)
    with open(folderpath+"/events.pkl") as f:
        rawEvents = f.readlines()

    #State of parsing
    lastWaslp :bool = False
    currentEvent:str = ""
    currentClip:int = 0

    #Expressions to check for
    clipNumberExp = re.compile("^p\d*")
    clipNameExp = re.compile("^aS.*")
    clipEventExp = re.compile("^asS.*")
    ambigousExp = re.compile("^S.*")
    lpExp = re.compile("^\(lp.*")

    sizeOfDataset  = 572
    # Results:
    events = np.empty(sizeOfDataset,dtype=object)
    clips = np.zeros((sizeOfDataset, 20, 360, 490))
    startTimes = np.zeros(sizeOfDataset)
    endTimes = np.zeros(sizeOfDataset)


    for line in rawEvents:
        if clipEventExp.match(line) or ambigousExp.match(line) and not lastWaslp:
            currentEvent = line.split("'")[1]
        elif clipNameExp.match(line) or ambigousExp.match(line) and lastWaslp:
            ##print("CurrentClip",currentClip)
            ##print(events)
            events[currentClip], startTimes[currentClip], endTimes[currentClip] =  readClipInfo(folderpath+"/"+line.split("'")[1]+'/')
            clips[currentClip] = loadPictures(folderpath+"/"+line.split("'")[1]+"/")
        elif clipNumberExp.match(line):
            currentClip = int(line[1:])
        else:
            #print("Something is strange width: "+line,file=sys.stderr)

        if lpExp.match(line):
            lastWaslp = True
        else:
            lastWaslp = False

    return  clips, events,startTimes,endTimes


if __name__ == '__main__':
    ##print(readCSVfolder("/home/henrik/Cogito/Hackathon/data/"))

    ##print(readTrackingBox("/home/henrik/Cogito/Hackathon/data/etgt5N2CSD8/clip_27/12_info.csv"))

    ##print(readClipInfo("/home/henrik/Cogito/Hackathon/data/_6MvD7aK_bI/clip_18/clip_info.csv"))

    #loadPictures("/home/henrik/Cogito/Hackathon/data/etgt5N2CSD8/clip_27/*.png", True)

    readData("/home/henrik/Cogito/Hackathon/data")
#
# #print("------")
# # X, labels = ImageUtils.read_images("/home/henrik/Cogito/Hackathon/data/etgt5N2CSD8/clip_27/*.png")
# # X = np.array([np.array(image) for image in X])
# # X = ImageUtils.numapy_grayscale(X)
#
#
# #X, labels = ImageUtils.read_images("data")
# #X = np.array([np.array(image) for image in X])
# #X = ImageUtils.numapy_grayscale(X)
#
# ##print(pictures[0])
# #print(grey)/