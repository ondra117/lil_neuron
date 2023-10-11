import youtube_dl
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from copy import deepcopy
import os
from time import sleep




class AudioExtractor:
    def __init__(self):
        ...

    def extract(self, url):
        while True:
            try:
                video_info = youtube_dl.YoutubeDL().extract_info(url = url,download=False)
            except youtube_dl.utils.DownloadError as e:
                print(f"Error: {e}")
                if "Sign in to confirm your age" in str(e):
                    return False
                print("Try again:")
                continue

            with open("music.json") as f:
                data = json.load(f)

            filename = f"music/{data['n_songs']}.%(ext)s"

            song = {
                "title":video_info["title"],
                "lyrics":"",
                "url":url
            }

            data["songs"][data["n_songs"]] = song

            options={
                "format":"bestaudio/best",
                "keepvideo":False,
                "outtmpl":filename,
                "postprocessor":[{
                    "key":"FFmpegExtractAudio",
                    "preferreddcodec":"wav"
                }]
            }
            try:
                with youtube_dl.YoutubeDL(options) as ydl:
                    ydl.download([video_info['webpage_url']])
                    # stream = ffmpeg.input(f"{data['n_songs']}.m4a")
                    # stream = ffmpeg.output(stream, f"{data['n_songs']}.wav")
            except youtube_dl.utils.DownloadError as e:
                print(f"Error: {e}")
                if "Sign in to confirm your age" in str(e):
                    return False
                print("Try again:")
                continue
            except youtube_dl.utils.ExtractorError as e:
                print(f"Error: {e}")
                if "Sign in to confirm your age" in str(e):
                    return False
                print("Try again:")
                continue

            sleep(1)
            os.chdir("music")
            os.system(f"ffmpeg -i {data['n_songs']}.m4a -ar 44000 {data['n_songs']}.wav")
            os.system(f"ffmpeg -i {data['n_songs']}.webm -ar 44000 {data['n_songs']}.wav")
            os.system(f"del {data['n_songs']}.m4a")
            os.system(f"del {data['n_songs']}.webm")
            os.chdir("..")

            print(f"Download complete... {filename}")

            data["n_songs"] += 1

            with open("music.json", "w") as f:
                json.dump(data, f, indent=4)
            break


if __name__ == '__main__':
    os.chdir("music")
    os.system(f"del *.part")
    os.chdir("..")
    ar = AudioExtractor()
    with open("links.json") as f:
        data = json.load(f)
    
    for link in deepcopy(data["front"]):
        if ar.extract(link) != False:
            data["done"].append(link)
        data["front"].pop()
        with open("links.json", "w") as f:
            json.dump(data, f, indent=4)

    # ar.end()