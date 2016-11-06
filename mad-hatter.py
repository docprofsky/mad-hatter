#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.arrowedLine(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import argparse
    from os import path

    parser = argparse.ArgumentParser(description="Place great hats above people's faces.")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-v', '--video-src', default=0)
    parser.add_argument('-i', '--input-file', type=path.abspath)
    parser.add_argument('-o', '--output-file', type=path.abspath)
    args = parser.parse_args()

    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    show_window = True

    if args.input_file is not None:
        args.video_src = args.input_file
        if args.output_file is not None:
            show_window = False
        else:
            args.video_src = 'synth:bg={}:size=800x600'.format(args.video_src)

    cam = create_capture(args.video_src, fallback='synth:bg=maga.png:size=640x480')

    # Load the image of the hat
    hatimg = cv2.imread('maga.png', cv2.IMREAD_UNCHANGED)

    # Create the mask for the hat
    hatmask = hatimg[:,:,3]

    # Create the mask for the transparent areas of the hat image
    hatmask_bg = cv2.bitwise_not(hatmask)

    # Remove the transparency from the hat image
    hatimg = hatimg[:,:,0:3]


    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        start_time = clock()
        rects = detect(gray, cascade)
        detect_time = clock() - start_time

        vis = img.copy()

        start_time = clock()
        for x1, y1, x2, y2 in rects:
            facewidth = x2 - x1

            # Scale the hat to the width of the face
            hatscale = (facewidth * 1.0) / hatimg.shape[1]
            scaledhat = cv2.resize(hatimg, None, fx=hatscale, fy=hatscale)

            hatheight, hatwidth = scaledhat.shape[0:2]

            mask = cv2.resize(hatmask, (hatwidth, hatheight))
            mask_inv = cv2.resize(hatmask_bg, (hatwidth, hatheight))

            # Calculate the cordinates of the top left and bottom right corners
            # of the hat area
            hatposx1 = int(round((x1 + facewidth / 2.0) - (hatwidth / 2.0)))
            hatposx2 = hatposx1 + hatwidth

            hatbottom = y1 + int(round(hatheight * 0.3))
            hattop = hatbottom - hatheight
            hatskip = 0

            # If the hat area goes off the top of the image set it to 0 and
            # remove the extra pixels from the top of the scaled hat image
            if hattop < 0:
                hatskip = -hattop
                scaledhat = scaledhat[hatskip:]
                hattop = 0

            # Get pixels from where the hat will be inserted
            hatarea = vis[hattop:hatbottom, hatposx1:hatposx2]

            # Get only the pixels from image where the hat image is transparent
            hatarea_bg = cv2.bitwise_and(hatarea, hatarea, mask=mask_inv[hatskip:])

            # Get the pixels of the hat image where the hat is
            hatarea_fg = cv2.bitwise_and(scaledhat, scaledhat, mask=mask[hatskip:])

            # Combine the where the hat is and is not
            hatarea_merged = cv2.add(hatarea_bg, hatarea_fg)

            # Put the hat where it belongs
            vis[hattop:hatbottom, hatposx1:hatposx2] = hatarea_merged


        hat_time = clock() - start_time

        if args.debug:
            draw_rects(vis, rects, (0, 255, 0))
            draw_str(vis, (20, 20), 'found {} faces in {:.1f} ms'.format(len(rects), detect_time * 1000))
            draw_str(vis, (20, 40), 'placed hats in {:.1f} ms'.format(hat_time * 1000))

        if show_window:
            cv2.imshow('mad-hatter', vis)

            if 0xFF & cv2.waitKey(5) == 27:
                break
        else:
            break

    if args.output_file is not None:
        cv2.imwrite(args.output_file, vis)

    cv2.destroyAllWindows()
