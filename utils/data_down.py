!pip install -q gdown
import os
import gdown
import zipfile

data_dir = "/content/data"
os.makedirs(data_dir, exist_ok=True)

file_id = "你的GoogleDrive檔案ID"
zip_path = os.path.join(data_dir, "dogs-vs-cats.zip")

if not os.path.exists(zip_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading dataset...")
    gdown.download(url, zip_path, quiet=False)
else:
    print("Dataset already downloaded.")

extract_dir = os.path.join(data_dir, "dogs-vs-cats")
if not os.path.exists(extract_dir):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
else:
    print("Dataset already extracted.")