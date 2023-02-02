import json

with open("links.json") as f:
        data = json.load(f)

with open("music.json") as f:
        music = json.load(f)

# print(len(data["done"]))

# print(len(music["songs"].keys()))

for key, url in zip(music["songs"].keys(), data["done"]):
    music["songs"][key]["url"] = url

with open("music.json", "w") as f:
    json.dump(music, f, indent=4)