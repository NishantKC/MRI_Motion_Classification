import os
from pathlib import Path

import cv2
import numpy as np

from Utils.DataUtils.LoadingUtils import readImage

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
baseDir = PROJECT_ROOT / "generated"
selectedDataPath = str(baseDir) + "/"

motionSeverities = ["M0/", "M1/", "M2/", "M3/", "M4/"]
imageNames = []
classElementsSize = []
for motionSeverity in motionSeverities:
    listdir = os.listdir(selectedDataPath + motionSeverity)
    classElementsSize.append(len(listdir))
    imageNames.append(listdir)

wholeIndex = list(np.cumsum(classElementsSize))
wholeIndex.insert(0, 0)


def getClasses():
    return ["without motion", "small motion", "mild motion", "moderate motion", "severe motion"]


def getImageAndClasses(i, show=False):
    selectedClass = 0
    for indexForEachClass, value in enumerate(wholeIndex):
        if value > i:
            selectedClass = indexForEachClass - 1
            break

    imageName = selectedDataPath + motionSeverities[selectedClass] + imageNames[selectedClass][i - wholeIndex[selectedClass]]
    image = readImage(imageName, show=False)
    # image = np.diff(image)
    image = cv2.resize(image, (256, 256))

    target = np.zeros((1, len(motionSeverities)))
    target[0, selectedClass] = 1
    return image, target[0]


def getLen():
    return wholeIndex[-1]
