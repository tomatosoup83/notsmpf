import os
import cv2
import shutil
import datetime
import splitVideo
import ppDetect
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
import numpy as np
from based_noise_blinks_detection import based_noise_blinks_detection
# make sure later u save the detected images into a folder

pathToVideo = "../eyeVids/pupil2.mov"
pathToLeft = "./videos/left_half.mp4"
pathToRight = "./videos/right_half.mp4"
confidenceThresh = 0.9
slider_min_conf = 0.75
slider_max_conf = 1.0
slider_step_conf = 0.05


# print stuff with timestamp at the start cuz it looks nice
# lmao
def dprint(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def splitEyes(video, left, right, widthThresh):
    dprint(f"attemping to convert video file '{video} into left and right videos '{left}' and '{right}'")
    # Paths
    input_video = video  # path to video
    output_left = left
    output_right = right

    splitVideo.split_video_left_right(input_video, output_left, output_right, widthThresh)



def resetFolder(folderName):
    if os.path.exists(folderName):
        dprint(f"folder '{folderName}' exists, removing contents in folder")
        try:
            shutil.rmtree(folderName)
            dprint(f"Folder '{folderName}' and all its contents deleted successfully.")
        except OSError as e:
            dprint(f"Error: {e}. An error occurred during deletion.")
        dprint(f"Making new '{folderName}'")
        os.makedirs(folderName)
    else: 
        dprint(f"Folder '{folderName}' does not exist, making the folder")
        try:
            os.makedirs(folderName)
        except OSError:
            dprint(f"Error: Creating folder '{folderName}'")


# split the video into multiple image files
def videoToImages(video, folderName):
    folderName = str(folderName)
    dprint(f"Trying to convert video '{video}' into frames and storing into '{folderName}'")
    # 2. convert the video into multiple .bmp files and store it in the tempImages folder
    cam = cv2.VideoCapture(video)
    currentframe = 0
    frameRate = cam.get(cv2.CAP_PROP_FPS)
    while True:
        ret,frame = cam.read()
        if ret:
            #name = './frames/' + folderName +'/frame' + str(currentframe) + '.bmp'
            name = os.path.join(folderName, 'frame' + str(currentframe) + '.bmp')
            dprint("Creating... " + name)

            cv2.imwrite(name, frame)

            currentframe += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()    
    dprint("All frames done!")

    return frameRate, currentframe

    # 3. turn images ito grayscale (actually i think this is part of the algorithm but meh)
    
def pupilDetectionInFolder(folderPath):
    dprint(f"Starting pupil detection in folder '{folderPath}'")
    conf = []
    diameter = []
    for i in range(len(os.listdir(folderPath))):
        filename = f"frame{i}.bmp"
        newPath = os.path.join(folderPath, filename)
        imgWithPupil, outline_confidence, pupil_diameter = ppDetect.detect(newPath)
        
        conf.append(outline_confidence)
        diameter.append(pupil_diameter)
        dprint(f"Showing image {newPath} with detected pupil...")
        # show the images continuously using cv2 window
        cv2.imshow("Pupil Detection for " + folderPath, imgWithPupil)
        cv2.waitKey(1)  # Display each image for 1 ms

        # closes window after all images are shown
    cv2.destroyAllWindows()
    return conf, diameter

def calculateTimeStamps(frameRate, totalFrames):
    timePerFrame = 1.0 / frameRate
    timestamps = [i * timePerFrame for i in range(totalFrames)]
    return timestamps






def blinkDetection(image):
    pass

# save data to csv with Columns: 'frame_id', 'timestamp', 'diameter', 'confidence', 'is_bad_data'
def saveDataToCSV(frameIDs, timestamps, diameters, confidences, outputPath):
    data = {
        'frame_id': frameIDs,
        'timestamp': timestamps,
        'diameter': diameters,
        'confidence': confidences
    }
    df = pd.DataFrame(data)
    # Mark bad data points (confidence < 1)
    df['is_bad_data'] = df['confidence'] < confidenceThresh
    df.to_csv(outputPath, index=False)
    dprint(f"Data saved to CSV at '{outputPath}'")
    
    # return the pandas dataframe too if needed
    return df
def plotResults(dataframe, savePath=None, showPlot=True, blink_intervals=None):
    # clear previous plots
    plt.clf()
    # plot data based on dataframe
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(dataframe['timestamp'], dataframe['diameter'], label='Pupil Diameter (pixels)', color='blue')
    if blink_intervals:
        for idx, (start_t, end_t) in enumerate(blink_intervals):
            label = 'Blink' if idx == 0 else None
            ax1.axvspan(start_t, end_t, color='orange', alpha=0.2, label=label)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Diameter (pixels)')
    ax1.set_title('Pupil Diameter Over Time')
    ax1.legend()
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(dataframe['timestamp'], dataframe['confidence'], label='Confidence', color='green')
    if blink_intervals:
        for idx, (start_t, end_t) in enumerate(blink_intervals):
            label = 'Blink' if idx == 0 else None
            ax2.axvspan(start_t, end_t, color='orange', alpha=0.2, label=label)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Confidence')
    ax2.set_title('Detection Confidence Over Time')
    ax2.legend()
    
    plt.tight_layout()
    # save the plot if a path is provided
    if savePath:
        plt.savefig(savePath)
        dprint(f"Plot saved to '{savePath}'")
    if showPlot:
        plt.show()

    

def preProcessData(df):
    # set rows with confidence < 1 to NaN
    dprint(f"Preprocessing data: setting diameters with confidence < {confidenceThresh} to NaN")
    df.loc[df['confidence'] < confidenceThresh, 'diameter'] = float('nan')
    return df

def compute_blink_intervals(dataframe, frame_rate, conf_threshold):
    sampling_freq = float(frame_rate)
    if sampling_freq <= 0:
        return [], np.zeros(len(dataframe), dtype=bool)

    sampling_interval = 1000.0 / sampling_freq
    pupil = np.nan_to_num(dataframe['diameter'].to_numpy(), nan=0.0).copy()
    confidence = dataframe['confidence'].to_numpy()
    pupil[np.where(confidence < conf_threshold)] = 0.0

    blinks = based_noise_blinks_detection(pupil, sampling_freq)
    onset_ms = np.asarray(blinks.get('blink_onset', []), dtype=float)
    offset_ms = np.asarray(blinks.get('blink_offset', []), dtype=float)

    intervals = []
    blink_mask = np.zeros_like(pupil, dtype=bool)

    for on_ms, off_ms in zip(onset_ms, offset_ms):
        start_t = on_ms / 1000.0
        end_t = off_ms / 1000.0
        intervals.append((start_t, end_t))

        start_idx = int(np.floor(start_t * sampling_freq))
        end_idx = min(len(blink_mask), int(np.ceil(end_t * sampling_freq)))
        blink_mask[start_idx:end_idx] = True

    return intervals, blink_mask


def plotResultsInteractive(dataframe, frame_rate):
    # Interactive plot with a confidence threshold slider controlling diameter masking
    timestamps = dataframe['timestamp'].to_numpy()
    confidence = dataframe['confidence'].to_numpy()
    diameter_raw = dataframe['diameter'].to_numpy()

    # Initial filtered diameter and blinks based on global confidenceThresh
    diameter_filtered = np.where(confidence < confidenceThresh, np.nan, diameter_raw)
    blink_intervals, _ = compute_blink_intervals(dataframe, frame_rate, confidenceThresh)

    fig, (ax_diam, ax_conf) = plt.subplots(2, 1, figsize=(12, 8))

    # Diameter plot (filtered by threshold)
    line_diam, = ax_diam.plot(timestamps, diameter_filtered, label='Pupil Diameter (filtered)', color='blue')
    blink_spans_diam = []
    for idx, (start_t, end_t) in enumerate(blink_intervals):
        label = 'Blink' if idx == 0 else None
        span = ax_diam.axvspan(start_t, end_t, color='orange', alpha=0.2, label=label)
        blink_spans_diam.append(span)
    ax_diam.set_xlabel('Time (s)')
    ax_diam.set_ylabel('Diameter (pixels)')
    ax_diam.set_title(f'Pupil Diameter Over Time (Thresh={confidenceThresh:.2f})')
    ax_diam.legend()

    # Confidence plot (static values) + threshold line
    ax_conf.plot(timestamps, confidence, label='Confidence', color='green')
    blink_spans_conf = []
    for idx, (start_t, end_t) in enumerate(blink_intervals):
        label = 'Blink' if idx == 0 else None
        span = ax_conf.axvspan(start_t, end_t, color='orange', alpha=0.2, label=label)
        blink_spans_conf.append(span)
    thresh_line = ax_conf.axhline(confidenceThresh, color='red', linestyle='--', label='Threshold')
    ax_conf.set_xlabel('Time (s)')
    ax_conf.set_ylabel('Confidence')
    ax_conf.set_title('Detection Confidence Over Time')
    ax_conf.legend()

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Slider for confidence threshold
    # Set initial axis limits for a more zoomed-out default view
    # Diameter axis padding based on raw diameter range
    try:
        ymin = np.nanmin(diameter_raw)
        ymax = np.nanmax(diameter_raw)
        pad = max(1.0, 0.10 * (ymax - ymin))
        ax_diam.set_xlim(timestamps[0], timestamps[-1])
        ax_diam.set_ylim(ymin - pad, ymax + pad)
    except Exception:
        # If all values are NaN or an error occurs, keep autoscale
        pass

    # Confidence axis range
    ax_conf.set_xlim(timestamps[0], timestamps[-1])
    ax_conf.set_ylim(0.0, 1.05)

    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Conf Thresh',
        valmin=slider_min_conf,
        valmax=slider_max_conf,
        valinit=float(np.clip(confidenceThresh, slider_min_conf, slider_max_conf)),
        valstep=slider_step_conf
    )

    def update(val):
        new_thresh = slider.val
        # update filtered diameter based on new threshold
        updated = np.where(confidence < new_thresh, np.nan, diameter_raw)
        line_diam.set_ydata(updated)
        # recompute blink intervals and redraw spans
        new_intervals, _ = compute_blink_intervals(dataframe, frame_rate, new_thresh)
        for span in blink_spans_diam:
            span.remove()
        for span in blink_spans_conf:
            span.remove()
        blink_spans_diam.clear()
        blink_spans_conf.clear()
        for idx, (start_t, end_t) in enumerate(new_intervals):
            label = 'Blink' if idx == 0 else None
            span_d = ax_diam.axvspan(start_t, end_t, color='orange', alpha=0.2, label=label)
            span_c = ax_conf.axvspan(start_t, end_t, color='orange', alpha=0.2, label=None)
            blink_spans_diam.append(span_d)
            blink_spans_conf.append(span_c)
        # update title and threshold line
        ax_diam.set_title(f'Pupil Diameter Over Time (Thresh={new_thresh:.2f})')
        thresh_line.set_ydata([new_thresh, new_thresh])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

# ENTRY POITN!!!

resetFolder("videos")
#splitEyes(pathToVideo, pathToLeft, pathToRight, 600)
resetFolder("frames")
#resetFolder("frames/left")
#resetFolder("frames/right")
#videoToImages(pathToLeft,"left")
#videoToImages(pathToRight,"right")
frameRate, totalFrames = videoToImages(pathToVideo,"frames")

#pupilDetectionInFolder("frames/left/")
#pupilDetectionInFolder("frames/right/")
conf, diameter = pupilDetectionInFolder("frames/")



timestamps = calculateTimeStamps(frameRate, totalFrames)

# make folder for data of this video
csvDataFolder = os.path.join("data", os.path.basename(pathToVideo).split('.')[0])
resetFolder(csvDataFolder)

csvDataPath = os.path.join(csvDataFolder, "raw_data.csvs")
df = saveDataToCSV(list(range(totalFrames)), timestamps, diameter, conf, csvDataPath)
# detect blinks on raw data using current confidence threshold
blink_intervals, blink_mask = compute_blink_intervals(df, frameRate, confidenceThresh)
df_raw = df.copy()
df_raw['is_blink'] = blink_mask

# plot raw data with blink overlays
plotResults(df_raw, savePath=os.path.join(csvDataFolder, "plotRaw.png"), showPlot=False, blink_intervals=blink_intervals)

# preprocess data
df = preProcessData(df)
df['is_blink'] = blink_mask
# save again after preprocessing
csvDataPathProcessed = os.path.join(csvDataFolder, "processed_data.csvs")
df.to_csv(csvDataPathProcessed, index=False)
# save processed static plot
plotResults(df, savePath=os.path.join(csvDataFolder, "plotProcessed.png"), showPlot=False, blink_intervals=blink_intervals)
# show interactive plot based on raw data (adjust threshold live)
plotResultsInteractive(df_raw, frameRate)