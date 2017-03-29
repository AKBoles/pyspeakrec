''' 
IMPORTANT: 
	-This script was adapted from the following to meet our needs.
	-The main changes made were needed with our new data set, most of functionality remains the same.
	-URL: https://github.com/pannous/tensorflow-speech-recognition/blob/master/speech_data.py
'''

import gzip
import os
import re
import numpy
import numpy as np
import wave
# import extensions as xx
from random import shuffle
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

# CLEAN THIS UP, DO NOT NEED ALL OF THIS
DATA_DIR = '/home/cc/Data/'
train_path = '/home/cc/Data/train-100-clean/' # 100 GB training data
CHUNK = 4096
test_fraction=0.1 # 10% of data for test / verification

# http://pannous.net/files/spoken_numbers_pcm.tar
class Source:  # labels
  NUMBER_WAVES = 'spoken_numbers_wav.tar'
  DIGIT_WAVES = 'spoken_numbers_pcm.tar'
  DIGIT_SPECTROS = 'spoken_numbers_spectros_64x64.tar'  # 64x64  baby data set, works astonishingly well
  NUMBER_IMAGES = 'spoken_numbers.tar'  # width=256 height=256
  SPOKEN_WORDS = 'https://dl.dropboxusercontent.com/u/23615316/spoken_words.tar'  # width=512  height=512# todo: sliding window!
  TEST_INDEX = 'test_index.txt'
  TRAIN_INDEX = 'train_index.txt'

from enum import Enum
class Target(Enum):  # labels
  digits=1
  speaker=2
  words_per_minute=3
  word_phonemes=4
  word=5#characters=5
  sentence=6
  sentiment=7
  first_letter=8

def speaker(file):  # vom Dateinamen
  # if not "_" in file:
  #   return "Unknown"
  return file.split("_")[0]
  #return file.split("_")[1]

# change this function to allow for input path
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
  # items=set(items) # assure uniqueness
  x=[0]*len(items)# numpy.zeros(len(items))
  i=items.index(item)
  x[i]=1
  return x

def dense_to_one_hot(batch, batch_size, num_labels):
  sparse_labels = tf.reshape(batch, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  concatenated = tf.concat(1, [indices, sparse_labels])
  concat = tf.concat(0, [[batch_size], [num_labels]])
  output_shape = tf.reshape(concat, [2])
  sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
  return tf.reshape(sparse_to_dense, [batch_size, num_labels])

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  return numpy.eye(num_classes)[labels_dense]

def extract_labels(names_file,train, one_hot):
  labels=[]
  for line in open(names_file).readlines():
    image_file,image_label = line.split("\t")
    labels.append(image_label)
  if one_hot:
      return dense_to_one_hot(labels)
  return labels

def extract_images(names_file,train):
  image_files=[]
  for line in open(names_file).readlines():
    image_file,image_label = line.split("\t")
    image_files.append(image_file)
  return image_files

if __name__ == "__main__":
  print('This is speech_data.py')
