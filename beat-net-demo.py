import sys
import os
from BeatNet.BeatNet import BeatNet
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.generators import Sine
from moviepy.editor import VideoFileClip

def extract_audio(input_file):
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext == '.mp4':
        video = VideoFileClip(input_file)
        audio = video.audio
        temp_audio_file = input_file.replace('.mp4', '_temp.wav')
        audio.write_audiofile(temp_audio_file)
        video.close()
        return temp_audio_file
    return input_file

def process_audio(input_file):
    # Get the directory, filename, and extension of the input file
    file_dir = os.path.dirname(os.path.abspath(input_file))
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    file_ext = os.path.splitext(input_file)[1].lower()

    # Extract audio if it's an MP4 file
    audio_file = extract_audio(input_file)

    # Load and process the audio file
    y, sr = librosa.load(audio_file)

    # Initialize BeatNet
    estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

    # Process the audio file
    Output = estimator.process(audio_file)

    # Extract beat and downbeat information
    beats = Output[:, 0]
    beat_types = Output[:, 1]

    # Determine the beat pattern
    unique_beat_types = np.unique(beat_types)
    if len(unique_beat_types) == 2:
        pattern = "1-2"
    elif len(unique_beat_types) == 4:
        pattern = "1-2-3-4"
    elif len(unique_beat_types) == 3:
        pattern = "1-2-3"
    else:
        pattern = "unknown"

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

    # Clean up temporary audio file if it was created
    if audio_file != input_file:
        os.remove(audio_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python beat-net-demo.py <path_to_audio_or_video_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    process_audio(input_file)
