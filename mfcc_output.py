# Beat tracking example
from __future__ import print_function

# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
from tqdm import *

import time

ms.use('seaborn-muted')

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display

import os

from PIL import Image

directory = os.path.dirname(os.path.realpath(__file__))
audio_path = os.path.join(directory,"audio")
mfcc_path = os.path.join(directory,"mfcc_output")
crop_path = os.path.join(directory,"crop")


def mfcc_plot(filename):
    plt.close()
    y, sr = librosa.load(os.path.join(audio_path,filename))
    S = librosa.feature.mfcc(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    plt.interactive(False)
    librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    fig = plt.gcf()
    basename = os.path.splitext(filename)[0]
    fig.savefig(mfcc_path + "/"+ basename + ".png", dpi=100)


def plot_crop(filename):
    img = Image.open(os.path.join(mfcc_path,filename))
    area = (80, 38, 804, 342)
    cropped_img = img.crop(area)
    basename = os.path.splitext(filename)[0]
    cropped_img.save(crop_path+"/"+basename+".png")


def mfcc_save():
    print("Saving MFCC ...")
    for file in tqdm(os.listdir(audio_path)):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            mfcc_plot(filename)
            time.sleep(.01)
        else:
            print("error")
    print("Successfully saved")

# Might have DS.Store hidden file on Mac OS X

def crop_save():
    print("Cropping MFCC ...")
    for file in tqdm(os.listdir(mfcc_path)):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            plot_crop(filename)
            time.sleep(.01)
        else:
            print("error on " + filename)
    print("Successfuly cropped")


crop_save()




####################################
'''
# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S = librosa.power_to_db(S, ref=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()
'''
######################################
