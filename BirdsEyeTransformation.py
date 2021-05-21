"""
BirdsEyeTransformation.py

This program converts the input video into a “bird’s eye” output video.
It converts a frame into the bird’s eye transformation, so that the road is rectangular

Give path to directory with images as input after running the program

@author: Anushree Das (ad1707)
"""

import cv2 as cv
from os import path
import numpy as np


def canny_edge_detector(image):
    # Convert the image color to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # Reduce noise from the image
    blur = cv.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)

    # Fill poly-function deals with multiple polygon
    cv.fillPoly(mask, polygons, 255)

    # Bitwise operation between canny image and mask image
    masked_image = cv.bitwise_and(image, mask)
    return masked_image


def create_coordinates(image, line_parameters):
    # print(line_parameters)
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = 0
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # It will fit the polynomial and the intercept and slope
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = create_coordinates(image, left_fit_average)
    right_line = create_coordinates(image, right_fit_average)
    return np.array([left_line, right_line]).astype('uint32')


def processVideo(filename):
    print('Processing video..', filename)

    # Create a capture object
    cap = cv.VideoCapture(cv.samples.findFileOrKeep(filename))

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    found = False

    # get coordinates of center lane to warp it later
    while not found:
        ret,frame = cap.read()
        # get edges
        canny_image = canny_edge_detector(frame)

        # keep only white lines of center lane
        cropped_image = region_of_interest(canny_image)

        # get coordinates for white lines of center lane
        lines = cv.HoughLinesP(cropped_image, 2, np.pi / 180, 100,
                               np.array([]), minLineLength=40,
                               maxLineGap=5)

        # if more than one line found
        if lines is not None and len(lines)>1:
            # extend the white lines of center lane to cover complete lane and get left and right line coordinates
            averaged_lines = average_slope_intercept(frame, lines)
            x1l, y1l, x2l, y2l = averaged_lines[0]
            x1r, y1r, x2r, y2r = averaged_lines[1]
            found = True

    # reset video
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    # Read until video is completed or 500 frames are processed
    while True:
        # if video file successfully open then read frame from video
        if (cap.isOpened):
            # to store list of frames to select brightest of them
            ret, frame = cap.read()

            # exit when we reach the end of the video
            if ret == 0:
                break

            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            warped_img = birdsEyeTransformation(frame, width, height, x1l, y1l, x2l, y2l,x1r, y1r, x2r, y2r)

            cv.imshow('video', warped_img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    # close the video file
    cap.release()

    # destroy all the windows that is currently on
    cv.destroyAllWindows()


def birdsEyeTransformation(frame, width, height, x1l, y1l, x2l, y2l,x1r, y1r, x2r, y2r):
    # the corners of the polygon created by lane edges
    src = np.float32([[x2r, y2r], [x2l, y2l],[x1l, y1l],[x1r, y1r] ])
    # the corners of rectangle to which the previous polygon should be mapped to
    dst = np.float32([[x2r, y2r], [x2l, y2l],[x2l, y1l],[x2r, y1r] ])

    # the transformation matrix
    M = cv.getPerspectiveTransform(src, dst)

    # bird’s eye transformation
    warped_img = cv.warpPerspective(frame, M, (width, height) ) # Image warping

    # resize image to quarter of the resolution
    dim = (int(0.25*width), int(0.25*height))
    resized = cv.resize(warped_img, dim, interpolation=cv.INTER_AREA)

    return resized


def main():
    # filename = 'road_videos/MVI_2201_CARS_ON_590_FROM_BRIDGE.MOV'
    # filename = 'road_videos/MVI_2207_CARS_ON_590_FROM_BRIDGE.MOV'
    # filename = 'road_videos/MVI_2208_CARS_ON_590_FROM_BRIDGE.MOV'

    # list of  video formats
    ext = [".mov"]

    # take filename or directory name from user
    filename = input('Enter filepath:')
    if path.exists(filename):
        _, file_extension = path.splitext(filename)
        if file_extension.lower() in ext:
            processVideo(filename)
        else:
            print("File isn't supported")
    else:
        print("File doesn't exist")


if __name__ == "__main__":
    main()
