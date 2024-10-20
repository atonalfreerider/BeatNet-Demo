import sys
import os
from BeatNet.BeatNet import BeatNet
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.generators import Sine

def process_audio(input_file):
    # Get the directory, filename, and extension of the input file
    file_dir = os.path.dirname(os.path.abspath(input_file))
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    file_ext = os.path.splitext(input_file)[1].lower()

    # Load and process the audio file
    y, sr = librosa.load(input_file)

    # Initialize BeatNet
    estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

    # Process the audio file
    Output = estimator.process(input_file)

    # Extract beat and downbeat information
    beats = Output[:, 0]
    downbeats = Output[Output[:, 1] == 2, 0]

    # Create the plot
    plt.figure(figsize=(20, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    for beat in beats:
        if beat in downbeats:
            plt.axvline(x=beat, color='r', linewidth=2, alpha=0.8)  # Downbeats (heavy hash mark)
        else:
            plt.axvline(x=beat, color='g', linewidth=1, alpha=0.5)  # Regular beats (light tick mark)
    plt.title('Audio Waveform with Beats and Downbeats')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f"{file_name}_beat-analysis.png"))
    plt.close()

    # Create a WAV file with metronome ticks for downbeats and offbeats
    audio = AudioSegment.from_file(input_file, format=file_ext[1:])
    downbeat_tick = Sine(1200).to_audio_segment(duration=50).fade_out(25).apply_gain(-3)
    offbeat_tick = Sine(800).to_audio_segment(duration=50).fade_out(25).apply_gain(-6)

    for beat in beats:
        position_ms = int(beat * 1000)
        if beat in downbeats:
            audio = audio.overlay(downbeat_tick, position=position_ms)
        else:
            audio = audio.overlay(offbeat_tick, position=position_ms)

    audio.export(os.path.join(file_dir, f"{file_name}_beat-analysis.wav"), format="wav")

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
    with open(os.path.join(file_dir, f"{file_name}_beat-analysis.json"), 'w') as json_file:
        json.dump(numpy_to_list(Output), json_file, indent=4)

    print(f"Analysis complete for {input_file}")
    print(f"Outputs saved in {file_dir}:")
    print(f"  - Waveform plot: {file_name}_beat-analysis.png")
    print(f"  - Audio with beats: {file_name}_beat-analysis.wav")
    print(f"  - Beat data: {file_name}_beat-analysis.json")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python beat-net-demo.py <path_to_audio_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    process_audio(input_file)
