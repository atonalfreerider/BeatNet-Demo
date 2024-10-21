import sys
import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.generators import Sine

def load_beat_data(beat_file):
    beats = []
    beat_types = []
    with open(beat_file, 'r') as f:
        for line in f:
            time, beat = map(float, line.strip().split())
            beats.append(time)
            beat_types.append(int(beat))
    return np.array(beats), np.array(beat_types)

def determine_beat_pattern(beat_types):
    unique_beat_types = np.unique(beat_types)
    if len(unique_beat_types) == 2:
        return "1-2"
    elif len(unique_beat_types) == 4:
        return "1-2-3-4"
    elif len(unique_beat_types) == 3:
        return "1-2-3"
    else:
        return "unknown"

def process_audio(audio_file, beat_file):
    # Get the directory, filename, and extension of the input file
    file_dir = os.path.dirname(os.path.abspath(audio_file))
    file_name = os.path.splitext(os.path.basename(audio_file))[0]

    # Load and process the audio file
    y, sr = librosa.load(audio_file)

    # Load beat data
    beats, beat_types = load_beat_data(beat_file)

    # Determine the beat pattern
    pattern = determine_beat_pattern(beat_types)

    # Create the plot
    plt.figure(figsize=(20, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    for beat, beat_type in zip(beats, beat_types):
        if beat_type == 1:
            plt.axvline(x=beat, color='r', linewidth=2, alpha=0.8)  # Downbeats (heavy hash mark)
        elif (pattern == "1-2" and beat_type == 2) or (pattern in ["1-2-3-4", "1-2-3"] and beat_type == 3):
            plt.axvline(x=beat, color='g', linewidth=1.5, alpha=0.6)  # Upbeats (medium hash mark)
        else:
            plt.axvline(x=beat, color='b', linewidth=1, alpha=0.4)  # Other beats (light hash mark)
    plt.title(f'Audio Waveform with Beats (Pattern: {pattern})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f"{file_name}_beat-analysis.png"))
    plt.close()

    # Create a WAV file with metronome ticks for different beat types
    audio = AudioSegment.from_file(audio_file)
    downbeat_tick = Sine(1000).to_audio_segment(duration=50).fade_out(25).apply_gain(-3)
    upbeat_tick = Sine(1200).to_audio_segment(duration=50).fade_out(25).apply_gain(-6)
    other_tick = Sine(800).to_audio_segment(duration=50).fade_out(25).apply_gain(-9)

    for beat, beat_type in zip(beats, beat_types):
        position_ms = int(beat * 1000)
        if beat_type == 1:
            audio = audio.overlay(downbeat_tick, position=position_ms)
        elif (pattern == "1-2" and beat_type == 2) or (pattern in ["1-2-3-4", "1-2-3"] and beat_type == 3):
            audio = audio.overlay(upbeat_tick, position=position_ms)
        else:
            audio = audio.overlay(other_tick, position=position_ms)

    audio.export(os.path.join(file_dir, f"{file_name}_beat-analysis.wav"), format="wav")

    # Prepare data for JSON output
    output_data = np.column_stack((beats, beat_types))

    # Save the output to JSON
    with open(os.path.join(file_dir, f"{file_name}_beat-analysis.json"), 'w') as json_file:
        json.dump(output_data.tolist(), json_file, indent=4)

    print(f"Analysis complete for {audio_file}")
    print(f"Outputs saved in {file_dir}:")
    print(f"  - Waveform plot: {file_name}_beat-analysis.png")
    print(f"  - Audio with beats: {file_name}_beat-analysis.wav")
    print(f"  - Beat data: {file_name}_beat-analysis.json")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python beat_this_analyzer.py <path_to_audio_file> <path_to_beat_data_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    beat_file = sys.argv[2]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        sys.exit(1)
    
    if not os.path.exists(beat_file):
        print(f"Error: Beat data file '{beat_file}' not found.")
        sys.exit(1)
    
    process_audio(audio_file, beat_file)
