import cv2
import os
import numpy as np

def read_video(videopath):
    """
    Reads a video file and returns its frames along with video properties.

    Args:
        videopath (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - frames (list): A list of frames read from the video.
            - frame_width (int): The width of the video frames.
            - frame_height (int): The height of the video frames.
            - frame_rate (float): The frame rate of the video.
    """

    #TODO: Complete this line to read the video file
    cap = cv2.VideoCapture(videopath) 
    
    #TODO: Check if the video was successfully opened
    if not cap.isOpened():
        print('Error: Could not open the video file')

    #TODO: Get the szie of frames and the frame rate of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Get the width of the video frames
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the height of the video frames
    frame_rate = cap.get(cv2.CAP_PROP_FPS) # Get the frame rate of the video
    
    #TODO: Use a loop to read the frames of the video and store them in a list
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_width, frame_height, frame_rate

if __name__ == "__main__":
    videopath = "../data/visiontraffic.avi"
    frames, frame_width, frame_height, frame_rate = read_video(videopath)