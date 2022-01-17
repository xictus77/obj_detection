# Source: https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# Source: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
# References: https://www.edureka.co/blog/python-opencv-tutorial/
# Source: https://github.com/mailrocketsystems/AIComputerVision

# Reference for drawing ROI: https://www.geeksforgeeks.org/python-draw-rectangular-shape-and-extract-objects-using-opencv/
# Reference https://github.com/Grad-CSE2016/Abandoned-Object-Detection/blob/master/AbandonedObjectDetection.py


# import the necessary packages
from imutils.video import VideoStream
import argparse
from datetime import datetime
import imutils
import time
import cv2
import os
import numpy as np
import sys
import pandas as pd
from tracker.centroidtracker import CentroidTracker
from readjson import read_json


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--server-ip",
#                 help="ip address of the server to which the client will connect")
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# else if args.get("video", RSTP_URL)
#     vs = VideoStream(src=RSTP_URL).start()
#     time.sleep(2.0)

# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])


cwd = os.getcwd()

# initialize the first frame in the video stream
# firstFrame = None
firstFrame = None
status_list = [None, None]
times = []
timer = []
df = pd.DataFrame(columns = ["Start","End", "duration_in_sec"])

fps_start_time = datetime.now()
fps = 0
total_frames = 0

object_id_list = []
dtime = dict()
dwell_time = dict()
allowable_duration = 10.0 # stipulated allowable duration for static obstruction in seconds

timestr = time.strftime("%H%M%S")
t = int(timestr)
filename = "sample_camera/file_%a.avi" % t
filename_d = "sample_camera/file_d_%a.avi" % t
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# variables for drawing ROI

points = [(919.2771084337348,417.8554216867469),
          (1472,1079),
          (818,1079),
          (787.9518072289156,884.1204819277108),
          (762.6506024096385,428.69879518072287)]

points_roi = polys
print(f'ROIS LOADED: \n{points_roi}')
"""
points = [(12.668639053254438, 281.6568047337278),
          (115.62721893491124, 197.0414201183432),
          (396.10059171597635, 193.49112426035504),
          (519.1775147928994, 485.79881656804736),
          (12.668639053254438, 495.85798816568047)]
"""
points_resized = (np.int32(np.int32(points)/2)) # rescale points as the frame is resized from 1920 px to 960 px

polygon = [np.int32(points)]



# initialize our centroid tracker and frame dimensions -- added for centroid tracker
ct = CentroidTracker()
(H, W) = (None, None)

# initialize colors
GREEN = (0 , 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0 ,0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
out = cv2.VideoWriter(filename, fourcc, 25.0, (960, 540))
out_det = cv2.VideoWriter(filename_d, fourcc, 25.0, (960, 540))

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

"""START DOCSTRING 
# MOUSE CLICK FUNCTION

def left_click_detect(event, x, y, flags, points):
    if (event == cv2.EVENT_LBUTTONDOWN):
        print(f"\tClick on {x}, {y}")
        points.append([x,y])
        print(points)

END DOCSTRING"""

# loop over the frames of the video
while True:
    # grab the current frame and initialize the text
    frame = vs.read()
    status = 0
    frame = frame if args.get("video", None) is None else frame[1]

    text = "CLEAR FROM OBSTRUCTION"
    textcolor = GREEN

    # if the frame could not be grabbed, then we have reached the end of the video
    if frame is None:
        break

    # drawing of ROI
    frame = cv2.polylines(frame, polygon, True, (255, 0, 0), thickness=1)

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=960)
    frame1 = frame.copy()
    total_frames = total_frames + 1

    """DOC STRING: DRAW POLYGON ROI

    if len(points) > 1:
        for i in range(len(points)-1):
            # draw lines if the number of clicked points larger than 1
            cv2.line(frame, tuple(points[i]), tuple(points[i+1]), GREEN, 5)
        if len(points) > 2:
            # draw polygon if the number of clicked points larger than 2
            cv2.polylines(frame, polygon, True, BLACK, 3)
            cv2.fillPoly(frame, polygon, WHITE, 1)

    frame = cv2.polylines(frame, polygon, True, (255, 0, 0), thickness=5)
    mask_2 = cv2.fillPoly(frame.copy(), polygon, (0, 0, 0))
    mask_2inv = cv2.bitwise_not(mask_2)

    show_image = cv2.addWeighted(src1=frame1, alpha=0.6, src2=mask_2, beta=0.4, gamma=0)
    ROI = cv2.bitwise_and(frame1, mask_2inv)


    END DOC STRING: DRAW POLYGON ROI"""

    # print(frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # # if the frame dimensions are None, grab them - added for centroid tracker
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # if the first frame is None, initialize it
    # This is used to store the first image / frame of the video
    if firstFrame is None:
        firstFrame = gray
        continue

    # compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # backSub = cv2.createBackgroundSubtractorMOG2()
    # fgMask = backSub.apply(frame)
    # frameDelta = cv2.absdiff(fgMask, gray)
    # thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]


    # dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    rects = [] # centroid tracker - List of bounding rectangles

    # cv2.polylines(frame, polygon, True, BLACK, 3)
    # frame = cv2.polylines(frame, polygon, True, (255, 0, 0), thickness=1)

        # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        status = 1


        # compute the bounding box for the contour, draw it on the frame and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        # print(cv2.boundingRect(c))
        # rects.append(cv2.boundingRect(c)) # centroid tracker
        rects.append((x, y, x + w, y + h)) # centroid tracker - determine the centroid from bounding rectangle
        # print(rects)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), RED, 2)
        text = "OBJECTS DETECTED"
        textcolor = WHITE

    boundingboxes = np.array(rects)
    boundingboxes = boundingboxes.astype(int)
    rects = non_max_suppression_fast(boundingboxes, 0.3)
    # print(rects)


    # update our centroid tracker using the computed set of bounding box rectangles
    objects = ct.update(rects)
    # print(objects)

    # loop over the tracked objects
    # for (objectID, centroid) in objects.items():
    #     # draw both the ID of the object and the centroid of the object on the output frame
    #     object_id = "ID {}".format(objectID)
    #     cv2.putText(frame, object_id, (centroid[0] - 10, centroid[1] - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # print("objectId", objectId)
        # print("bbox", bbox)

        # Check if bbox is in ROI and return a value 1 if inside, 0 if on contour, -1 if outside
        # in_roi_id = cv2.pointPolygonTest(np.int32(points), (x1, y1), False)
        in_roi_id = cv2.pointPolygonTest(points_resized, (x1, y1), False)
        # print (x1, y1)
        # print(in_roi_id, x1, y1)

        if objectId not in object_id_list:
            object_id_list.append(objectId)
            dtime[objectId] = datetime.now()
            dwell_time[objectId] = 0
        else:
            curr_time = datetime.now()
            old_time = dtime[objectId]
            time_diff = curr_time - old_time
            dtime[objectId] = datetime.now()
            sec = time_diff.total_seconds()
            # min = sec * 60
            dwell_time[objectId] += sec

            # if int(dwell_time[objectId]) > 10:
            #     cv2.imwrite('detected'+str(objectId)+'.jpeg', frame)

        # cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
        # text_id_time = "ID:{}| Time:{}".format(objectId, int(dwell_time[objectId]))
        # cv2.putText(frame, text_id_time, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, RED, 1)


        if dwell_time[objectId] >= allowable_duration and in_roi_id >= 0:

            cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
            text = "ALERT! STATIC OBSTRUCTION DETECTED"
            textcolor = RED
            text_id_time = "ID:{}| Time:{}".format(objectId, int(dwell_time[objectId]))
            cv2.putText(frame, text_id_time, (x2, y2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, RED, 1)



            cv2.imwrite('detected_ID_'+str(objectId)+'.jpeg', frame)
            out_det.write(frame)

    # Calculating Frame per sec and putting on frame
    """ DOC STRING : Calculating Frame per sec and putting on frame
    fps_end_time = datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1)
    """





    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())

        # name = "frame%d.jpg" %count
        # cv2.imwrite(str(datetime.now()) + '_' + name, frame)
        # count +=1

    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 2)
    cv2.putText(frame, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, BLACK, 1)





    # show the frame and record
    cv2.imshow("Security Feed", frame)
    # cv2.imshow("show_img", show_image)
    # cv2.setMouseCallback('Frame', left_click_detect, points)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    # cv2.imshow("ROI", ROI)
    # Record
    out.write(frame)

    key = cv2.waitKey(30) & 0xFF

    # if key == ord('s'):
    #     cv2.imwrite("detected.jpg", frame)
    #     print("frame saved")

    # if the `q` key is pressed, break from the loop
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed
    # elif key == ord('o'):
    #     polygon = [np.int32(points)]
    #     points = []


    # cv2.setMouseCallback('Security Feed', left_click_detect, points)


print(status_list)
print(times)
# print(len(times))

for i in range(0, len(times)-1):
    diff_t = ((times[i+1] - times[i]).total_seconds())
    df=df.append({"Start" : times[i], "End": times[i+1], "duration_in_sec": diff_t}, ignore_index=True)

print(df)
df.to_csv("Times.csv")

print(filename)
path = os.path.abspath(filename)
print(path)
print("Current Directory:", cwd)

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
out.release()
cv2.destroyAllWindows()







"""
def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 10, (0, 255, 255), 10)
        if "points" not in globals():
            points = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(points)
"""