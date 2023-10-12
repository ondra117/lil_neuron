from pytube import YouTube
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import json
from copy import deepcopy
import os
from time import sleep
import ffmpeg

output_folder = "music/"

class AudioExtractor:
    def __init__(self):
        ...

    def extract(self, url, n):
        filename = f"{n}.".zfill(6)
        try:
            yt = YouTube(url)

            audio_stream = yt.streams.filter(only_audio=True).first()

            audio_stream.download(output_path=output_folder, filename=f"{filename}mp4")
        except Exception as e:
            print(e)
            return False
        mp4_file = os.path.join(output_folder, f"{filename}mp4")
        wav_file = os.path.join(output_folder, f"{filename}wav")

        ffmpeg.input(mp4_file).output(wav_file, ar=44000, ac=1).run(overwrite_output=True)
        os.remove(mp4_file)

        return True

if __name__ == '__main__':
    ar = AudioExtractor()
    with open("links.json") as f:
        data = json.load(f)
    
    for link in deepcopy(data["front"]):
        if ar.extract(link, len(data["done"])):
            data["done"].append(link)
        data["front"].pop(0)
        with open("links.json", "w") as f:
            json.dump(data, f, indent=4)

    # ar.end()