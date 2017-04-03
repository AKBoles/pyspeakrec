import os
import sys
import librosa
import pydub
import pyaudio
import tflearn
import audio_manip

'''
Need to figure out how to make this a function.
'''

# for now - directories
training_dir = sys.argv[1]
training_seg = sys.argv[2]

def create_net(num_layers, dropout, size, learning_rate, activation):
    # need to create input for shape of data
    # is this even necessary??

# train model over training directory, make sure to segment data
# if there is nothing in training_dir, need to record audio before segmenting?
if not os.listdir(training_dir):
    audio_manip.enroll(num_speakers=2, num_files=4, directory=training_dir)

length = 1
if not os.listdir(training_seg):
    audio_manip.segment(data=training_dir, seg_location=training_seg, length=length)

# create mfcc, delta, delta2 input features
hop_length = 128 # default = 512
X = []
Y = []
speakers = audio_manip.get_speakers(training_dir)
for f in os.listdir(training_seg):
    Y.append(audio_manip.one_hot_from_item(audio_manip.speaker(f), speakers))
    y, sr = librosa.load(training_seg + f)
    X.append(audio_manip.mfcc(y=y, sr=sr, num_mfcc=20, hop_length=128, delta=2)

# define the network for training
layer_size = 128
dropout = 0.7
learning_rate = 0.001
tflearn.init_graph(num_cores=8,gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 20, 519])
net = tflearn.fully_connected(net, layer_size, activation='relu')
net = tflearn.fully_connected(net, layer_size, activation='relu')
net = tflearn.fully_connected(net, layer_size, activation='relu')
net = tflearn.dropout(net, dropout)
net = tflearn.fully_connected(net, len(speakers), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=learning_rate)

# now train the model!
run_id='Demonstration'
model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=50, show_metric=True, snapshot_step=1000, run_id=run_id, validation_set=0.1)

# save the model to model_dir
model.save(model_dir + 'demo.model')
