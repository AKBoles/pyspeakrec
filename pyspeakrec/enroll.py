import os
import sys
import librosa
import pydub
import pyaudio
import wave
import tflearn
import time
from sys import byteorder
from array import array
from struct import pack 

#constants
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

def press_enter():
    while True:
        raw_input('Press enter to stop recording.')
        break
    return True

def is_silent(snd_data):
    # Returns 'True' if below the 'silent' threshold
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    # Average the volume out
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    # Trim the blank spots at the start and end
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

def add_silence(snd_data, seconds):
    # Add silence to the start and end of 'snd_data' of length 'seconds' (float)
    r = array('h', [0 for i in xrange(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in xrange(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.2 seconds of 
    blank sound.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True
        # this controls the length of recording?
        if snd_started and press_enter():
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5) # does this have to be 0.5??
    return sample_width, r

def record_to_file(path):
    # Records from the microphone and outputs the resulting data to 'path'
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

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
            t0 = time.time()
            record_to_file(directory + 'Speaker' + str(count) + '_' + str(s) + '.wav')
            print('Time during recording: %s' %str(time.time() - t0))
        count = count + 1

if __name__ == '__main__':
    num_speakers = int(sys.argv[1])
    num_files = int(sys.argv[2])
    directory = sys.argv[3]
    enroll(num_speakers=num_speakers, num_files=num_files, directory=directory)
