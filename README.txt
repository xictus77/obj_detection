This repository is an attempt to use object detection and tracking in a real world use case.

This post shall discuss on a use case where we attempt to use object detection and tracking algorithm to check for obstruction within a specified zone in a video feed using OpenCV-Python. The concept described is similar to that of an abandoned-object detection where we detect object and track object to determine static object and flagging it out for a prolonged period of time

example of use:

python obstruct_detect.py -v /path to the video file -a default=500

Output:
Snapshot of detected object in ROI
Video showing the detected object in ROI

Medium post:
https://xictus77.medium.com/obstruction-detection-and-tracking-using-opencv-python-ea5838822945