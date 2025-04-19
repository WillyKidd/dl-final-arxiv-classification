import os
import zipfile

import requests

url = "https://github.com/ashfarhangi/Protoformer/raw/refs/heads/main/data/ArXiv-10.zip"
zip_path = "ArXiv-10.zip"
extract_dir = "data"

os.makedirs(extract_dir, exist_ok=True)

print(f"Downloading {url}...")
response = requests.get(url)
with open(zip_path, "wb") as f:
    f.write(response.content)
print("Download complete.")

print(f"Extracting to ./{extract_dir}...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
print("Extraction complete.")

os.remove(zip_path)
