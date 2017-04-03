import os
import sys
import time
from pydub import AudioSegment as audio

# need to: create mfcc function, train/test set functions

# compute mfcc features using librosa
def mfcc(y, sr, num_mfcc, hop_length, delta):
    if delta = 0:
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, hop_length=hop_length)
    elif delta = 1:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, hop_length=hop_length)
        delta = librosa.feature.delta(mfcc)
        return np.c_[mfcc,delta]
    else:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, hop_length=hop_length)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return np.c_[mfcc, delta, delta2]

# segment data function
def segment(data, seg_location, length):
  os.chdir(data)
  files = os.listdir(data)
  speakers = speech_data.get_speakers(data)
  waves = []
  num = {}
  for s in speakers:
    num[s] = 0
  c = 0
  for f in files: # grab all wave files in list
    waves.append(audio.from_wav(f))
    c = c + 1
  os.chdir(seg_location)
  for f,w in zip(files,waves): # need to segment the data into one second intervals
    begin = 0
    end = 1
    while (end*length) < int(w.duration_seconds):
      segment = w[begin*1000*length:end*1000*length]
      segment.export(speech_data.speaker(f) + '_' +  str(num[speech_data.speaker(f)]) + '.wav', 'wav')
      begin = begin + length
      end = end + length
      num[speech_data.speaker(f)] = num[speech_data.speaker(f)] + 1

# get speaker name from file
def speaker(file):
  return file.split("_")[0]
  #return file.split("_")[1]

# get speakers for labels in a directory
def get_speakers(path):
  files = os.listdir(path)
  print('number of files: %s' %len(files))
  def nobad(file):
    return "_" in file and not "." in file.split("_")[0]
  speakers=list(set(map(speaker,filter(nobad,files))))
  print(len(speakers)," speakers: ",speakers)
  return speakers

def one_hot_to_item(hot, items):
  i=np.argmax(hot)
  item=items[i]
  return item

def one_hot_from_item(item, items):
  x=[0]*len(items)
  i=items.index(item)
  x[i]=1
  return x
