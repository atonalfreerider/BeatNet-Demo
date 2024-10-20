from BeatNet.BeatNet import BeatNet
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.generators import Sine

# Load and process the audio file
audio_file = "/home/john/Desktop/Audio/BeatNet-Demo/william-larissa.mp3"
y, sr = librosa.load(audio_file)

# Initialize BeatNet
estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

# Process the audio file
Output = estimator.process(audio_file)

# Function to safely extract beat information
def extract_beats(output):
    if isinstance(output, np.ndarray):
        if output.ndim == 2 and output.shape[1] == 2:
            return output[:, 0], output[output[:, 1] == 2, 0]
    return [], []

# Extract beat and downbeat information
beats, downbeats = extract_beats(Output)

# Create the plot
plt.figure(figsize=(20, 6))

# Plot the waveform
librosa.display.waveshow(y, sr=sr, alpha=0.5)

# Plot beats and downbeats
for beat in beats:
    if beat in downbeats:
        plt.axvline(x=beat, color='r', linewidth=2, alpha=0.8)  # Downbeats (heavy hash mark)
    else:
        plt.axvline(x=beat, color='g', linewidth=1, alpha=0.5)  # Regular beats (light tick mark)

plt.title('Audio Waveform with Beats and Downbeats')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig('waveform_with_beats.png')
plt.close()

# Create a WAV file with metronome ticks for downbeats and offbeats
audio = AudioSegment.from_mp3(audio_file)
downbeat_tick = Sine(1200).to_audio_segment(duration=50).fade_out(25).apply_gain(-3)
offbeat_tick = Sine(800).to_audio_segment(duration=50).fade_out(25).apply_gain(-6)

for beat in beats:
    position_ms = int(beat * 1000)
    if beat in downbeats:
        audio = audio.overlay(downbeat_tick, position=position_ms)
    else:
        audio = audio.overlay(offbeat_tick, position=position_ms)

audio.export("audio_with_beats.wav", format="wav")

# Convert numpy arrays to lists for JSON serialization
def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    return obj

# Save the output to JSON
with open('output.json', 'w') as json_file:
    json.dump(numpy_to_list(Output), json_file, indent=4)

print("Waveform with beats has been saved as 'waveform_with_beats.png'")
print("Audio with beat ticks has been saved as 'audio_with_beats.wav'")
print("BeatNet output has been saved as 'output.json'")
