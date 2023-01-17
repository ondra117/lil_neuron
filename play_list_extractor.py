from pytube import Playlist
import json

def get_playlist(playlist):
    with open("links.json") as f:
        data = json.load(f)

    act_links = set(data["done"] + data["front"])
    number = data["n"]

    urls = []
    playlist_urls = Playlist(playlist)
    for url in playlist_urls:
        if not url in act_links:
            number += 1
            print(f"new song added: {number}")
            urls.append(url)

    data["n"] = number
    data["front"] += urls

    with open("links.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    get_playlist("https://www.youtube.com/playlist?list=PLm0E3dYqEEugGjHHjy0SRYTLyVhzjkuBd")
