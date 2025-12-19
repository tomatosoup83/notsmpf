import os
import cv2
import shutil
import datetime
# make sure later u save the detected images into a folder

pathToVideo = "../videos/pupil_video_2.mp4"
# split the video into multiple image files

# print stuff with timestamp at the start cuz it looks nice
# lmao
def dprint(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def videoToImages(video):
    dprint(f"Trying to convert video '{video}' into images")
    # 1. clear the images folder
    #   hmm maybe delete the folder and make a new one everytime??

    if os.path.exists('frames'):
        dprint("folder for frames exists, removing contents in folder")
        try:
            shutil.rmtree('frames')
            dprint(f"Folder '{'frames'}' and all its contents deleted successfully.")
        except OSError as e:
            dprint(f"Error: {e}. An error occurred during deletion.")
        dprint("Making the new frames folder")
        os.makedirs('frames')
    else: 
        dprint("Folder frames does not exist, making the folder")
        try:
            os.makedirs('frames')
        except OSError:
            dprint("Error: Creating folder for images")
    # 2. convert the video into multiple .bmp files and store it in the tempImages folder
    cam = cv2.VideoCapture(pathToVideo)
    currentframe = 0

    while True:
        ret,frame = cam.read()
        if ret:
            name = './frames/frame' + str(currentframe) + '.jpg'
            dprint("Creating... " + name)

            cv2.imwrite(name, frame)

            currentframe += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()    
    dprint("All frames done!")
    # 3. turn images ito grayscale (actually i think this is part of the algorithm but meh)
    

def blinkDetection(image):
    pass

videoToImages(pathToVideo)