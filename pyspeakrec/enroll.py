import os
import sys
import librosa
import pydub
import pyaudio
import tflearn
home_dir = '/home/cc/pyspeakrec/'
sys.path.append(home_dir + 'Data/modules/')
import speech_data
import segment_data
import record_audio

'''
Should not need these here anymore!!
# constants - directory, etc.
# imput these according to operating system
training_dir = home_dir + 'Data/train/audio/'
training_seg = home_dir + 'Data/train/segment/'
testing_dir = home_dir + 'Data/test/audio/'
testing_seg = home_dir + 'Data/test/segment/'
model_dir = home_dir + 'Data/model/'
'''

# record audio for training, specify how many speakers and files per speaker with command line
# need to fix the time per recording - play with record_audio.record_to_file function
# maybe: Enter to begin, Enter to end?

def enroll(num_speakers, num_files, directory):
    count = 0
    while count < num_speakers:
        # record audio for each speaker - input strings for their names later
        for s in range(num_files):
            # record audio, save to a file
            raw_input('Press Enter to record an audio file.')
            record_audio.record_to_file(directory + 'Speaker' + str(count) + '_' + str(s) + '.wav')
        count = count + 1

if __name__ == '__main__':
    num_speakers = int(sys.argv[1])
    num_files = int(sys.argv[2])
    directory = sys.argv[3]
    enroll(num_speakers=num_speakers, num_files=num_files, directory=directory)
