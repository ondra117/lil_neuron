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
    CHROMDRIVER = "C:\Program Files (x86)\chromedriver.exe"
    def __init__(self):
        self.options = Options()
        self.options.headless = True
        self.driver = webdriver.Chrome(self.CHROMDRIVER, chrome_options=self.options)


    def get_lyrics(self, title):
        try:
            self.driver.get("https://www.google.com/")
            self.driver.find_element(By.ID, 'L2AGLb').click()
            search = self.driver.find_element_by_css_selector(".gLFyf")
            search.send_keys(f"{title} lyrics")
            search.send_keys(Keys.ENTER)
            lyrics = self.driver.find_elements_by_css_selector(".Z1hOCe .ujudUb span")
            lyrics = "\n".join(list(map(lambda x: x.text, lyrics)))
            return lyrics
        except Exception as e:
            return ""

    def end(self):
        self.driver.quit()

    def extract(self, url: str):
        while True:
            try:
                video_info = youtube_dl.YoutubeDL().extract_info(url = url,download=False)
            except youtube_dl.utils.DownloadError as e:
                print(f"Error: {e}")
                print("Try again:")
                continue

            with open("music.json") as f:
                data = json.load(f)

            filename = f"music/{data['n_songs']}.%(ext)s"

            song = {
                "title":video_info["title"],
                "lyrics":self.get_lyrics(video_info["title"]),
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
                print("Try again:")
                continue
            except youtube_dl.utils.ExtractorError as e:
                print(f"Error: {e}")
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
        ar.extract(link)
        data["front"].pop()
        data["done"].append(link)
        with open("links.json", "w") as f:
            json.dump(data, f, indent=4)

    ar.end()