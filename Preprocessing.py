# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:30:10 2022

@author: ljeantet
"""


from Preprocessing_helper_lj import *
import time

#parameters

species_folder = 'D:/Lemurs_2022/Lemurs' # Should contain /Audio and /Annotations
lowpass_cutoff = 4000 # Cutt off for low pass filter
downsample_rate = 9600 # Frequency to downsample to
nyquist_rate = 4800 # Nyquist rate (half of sampling rate)
segment_duration = 4 # how long should a segment be
augmentation_amount_positive_class = 1 # how many times should a segment be augmented
augmentation_amount_negative_class = 1 # how many times should a segment be augmented
positive_class = ['roar',] # which labels should be bundled together for the positive  class
negative_class = ['no-roar'] # which labels should be bundled together for the negative  class
file_type = 'svl'
audio_extension = '.wav'
n_fft = 1024 # Hann window length
hop_length = 256 # Sepctrogram hop size
n_mels = 128 # Spectrogram number of mells
f_min = 500 # Spectrogram, minimum frequency for call
f_max = 9000 # Spectrogram, maximum frequency for call
file = "TrainingFiles.txt" #txt file containing the name of the files to use for the training

pre_pro = Preprocessing(species_folder, lowpass_cutoff, 
                downsample_rate, nyquist_rate, 
                segment_duration,
                positive_class, negative_class, 
                augmentation_amount_positive_class, 
                augmentation_amount_negative_class,file, n_fft, 
                hop_length, n_mels, f_min, f_max, file_type, 
                audio_extension)


starting_time = time.time()

X_calls, Y_calls = pre_pro.create_dataset(True)


duration=time.time()-starting_time


print("Processing took {:.2f} seconds".format(duration))
print("Which is {:.2f} minutes".format(duration/60))


pre_pro.save_data_to_pickle(X_calls, Y_calls, "X_train", "Y_train")

