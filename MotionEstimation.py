"""
MotionEstimation.py

This program detects vehicles in the center lane of the road using the background subtraction
and stores the frame numbers in which a new car was detected.

Give path to directory with images as input after running the program

@author: Anushree Das (ad1707)
"""

import cv2 as cv
from os import path
import numpy as np

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(320, height), (850, height),(600,0),(550,0)]
    ])
    mask = np.zeros_like(image)

    # Fill poly-function deals with multiple polygon
    cv.fillPoly(mask, polygons, 255)

    # Bitwise operation between canny image and mask image
    masked_image = cv.bitwise_and(image, mask)
    return masked_image


def countCars(filename):
    print('Processing video..', filename)
    # create MOG2 BackgroundSubtractor object
    mog = cv.createBackgroundSubtractorMOG2(
        history=200, varThreshold=16, detectShadows=True)

    # Create a capture object
    cap = cv.VideoCapture(cv.samples.findFileOrKeep(filename))

    # keep count of frames processed for building background model
    counter = 0

    print('Building background model..')
    # Read until video is completed or 500 frames are processed
    while True:
        counter += 1
        # if video file successfully open then read frame from video
        if (cap.isOpened):

            ret, frame = cap.read()
            # exit when we reach the end of the video or 1000 frames are processed
            if ret == 0 or counter > 1000:
                break

        # build background model using the brightest frame
        mog.apply(frame)


    # get name of video
    name, file_extension = path.splitext(filename)

    # # get background model
    # bgmodel = mog.getBackgroundImage()
    # # save background model
    # cv.imwrite(name + 'background.jpg', bgmodel)

    f = open(name+'.csv','w')

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    prev_count = 0
    frame_counter = 0

    print('Detecting cars..')
    # Read until video is completed or 500 frames are processed
    while True:
        # if video file successfully open then read frame from video
        if (cap.isOpened):
            frame_counter += 1
            # to store list of frames to select brightest of them
            ret, frame = cap.read()
            # exit when we reach the end of the video
            if ret == 0:
                break

            fgmask = mog.apply(frame)
            # crop mask to get objects in the center lane
            cropped_image = region_of_interest(fgmask)

            # apply image erosion and dilation
            erosion = cv.erode(cropped_image, np.ones((3, 3), np.uint8), iterations=3)
            dilated = cv.dilate(erosion, np.ones((30, 30), np.uint8), iterations=2)

            # get width and height of the frame
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            cv.line(dilated, (0, 200), (width, 200), (200, 0, 0))

            # get contours from mask to get positions of the vehicles
            contours, hierarchy = cv.findContours(dilated.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            valid_cntrs = []

            # get valid contours for vehicles that didn't cross a certain line(line: y = 200)
            for i, cntr in enumerate(contours):
                x, y, w, h = cv.boundingRect(cntr)
                if (x <= width) & (y >= 200) & (cv.contourArea(cntr) >= 35000):
                    valid_cntrs.append(cntr)

            # number of discovered contours
            curr_count = len(valid_cntrs)

            # if number of detected vehicles in current frame is less than that of the last frame
            # then a vehicle has crossed the line(y=200)
            if curr_count < prev_count:
                print('Vehicle detected at frame: ',frame_counter)
                f.write(str(frame_counter)+'\n')

            prev_count = curr_count

    # close csv file
    f.close()
    # close the video file
    cap.release()

    # destroy all the windows that is currently on
    cv.destroyAllWindows()


def main():
    # filename = 'road_videos/MVI_2208_CARS_ON_590_FROM_BRIDGE.MOV'

    # list of  video formats
    ext = [".mov"]

    # take filename or directory name from user
    filename = input('Enter filepath::')

    if path.exists(filename):
        _, file_extension = path.splitext(filename)
        if file_extension.lower() in ext:
            # get name of video
            name, file_extension = path.splitext(path.basename(filename))
            if name == 'MVI_2208_CARS_ON_590_FROM_BRIDGE':
                countCars(filename)
            else:
                print('Incorrect file name. Required file: MVI_2208_CARS_ON_590_FROM_BRIDGE.mov')
        else:
            print("File isn't supported")
    else:
        print("File doesn't exist")


if __name__ == "__main__":
    main()
