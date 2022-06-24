import numpy as np
import os


def getVideoList(datasetPath):
    """
    Args:
        datasetPath (str): The path of UCF-Crime
    Returns:
        videoList: Returns a list of video absolute paths. Length is 1900.
    """
    videoList = []
    for curDir, dirs, files in os.walk(datasetPath):
        for file in files:
            if file.endswith('.mp4'):
                # videoList.append(file)
                videoList.append(os.path.join(curDir, file))
    return videoList

    

if __name__ == "__main__":

    datasetPath = '/home/jing/project/dataset/UCF-Crime-unzip/'
    videoList = getVideoList(datasetPath)
    # print(videoList)
    # print(len(videoList))

    
    


    
    

