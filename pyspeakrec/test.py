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

# constants - directory, etc.
# imput these according to operating system
training_dir = home_dir + 'Data/train/audio/'
training_seg = home_dir + 'Data/train/segment/'
testing_dir = home_dir + 'Data/test/audio/'
testing_seg = home_dir + 'Data/test/segment/'
model_dir = home_dir + 'Data/model/'

# record for testing model - for now just doing one speaker test file, later create another python script for this
# record audio, save to a file
raw_input('Press Enter to record an audio file.')
record_audio.record_to_file(testing_dir + 'test.wav')

# segment the recorded test audio
if not os.listdir(testing_seg):
    segment_data.segment(data=testing_dir, seg_location=testing_seg, length=length)

# create mfcc, delta, delta2 for testing data
Xtest = []
for f in os.listdir(testing_seg):
    y, sr = librosa.load(testing_seg + f)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    Xtest.append(np.c_[mfcc,delta,delta2])

# load the trained model from model_dir
model = DNN() # make sure this is correct
model.load(model_dir + 'demo.model') 

result = model.predict(Xtest)
c = 0
for f,r in zip(os.listdir(testing_seg), result):
    res = speech_data.one_hot_to_item(r, speakers)
    if res in f:
        c = c + 1
acc = float(c) / float(len(Xtest))
print('Test set accuracy: %s' %str(acc))

# need to add a part that outputs the speaker to command line
