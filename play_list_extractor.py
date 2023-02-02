from pytube import Playlist
import json

def get_playlist(playlist):
    with open("links.json") as f:
        data = json.load(f)

    act_links = set(data["done"] + data["front"])
    number = data["n"]

    playlist_urls = set(Playlist(playlist))
    playlist_urls -= act_links
    number += len(playlist_urls)

    data["n"] = number
    data["front"] += list(playlist_urls)

    with open("links.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    get_playlist("https://www.youtube.com/playlist?list=PLm0E3dYqEEugGjHHjy0SRYTLyVhzjkuBd")
