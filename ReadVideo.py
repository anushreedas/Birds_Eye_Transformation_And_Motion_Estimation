"""
ReadVideo.py

This program reads the first 30 frames of a video, and saves the thirtieth (30th) frame to a file.

Give path to directory with images as input after running the program

@author: Anushree Das (ad1707)
"""
import cv2 as cv
from os import path


def getFrames(filename):
    print('Processing video..', filename)

    # Create a capture object
    cap = cv.VideoCapture(cv.samples.findFileOrKeep(filename))

    cap.set(cv.CAP_PROP_POS_FRAMES, 30)
    # Read until video is completed or 500 frames are processed
    if (cap.isOpened):
            # to store list of frames to select brightest of them
            ret, frame = cap.read()
            if ret != 0:
                cv.imwrite(filename + "_frame_30.jpg", frame)
            else:
                print("Video doesn't have 30 frames" )

def main():

    # list of  video formats
    ext = [".mov"]

    # take filename or directory name from user
    filename = input('Enter filepath:')
    if path.exists(filename):
        _, file_extension = path.splitext(filename)
        if file_extension.lower() in ext:
            getFrames(filename)
        else:
            print("File isn't supported")
    else:
        print("File doesn't exist")


if __name__ == "__main__":
    main()
