import urllib.request
import tarfile
import os
import shutil
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    print(f"Downloading from: {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    tar_path = "data/YouTubeClips.tar"
    url = "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"
    
    # 1. Download
    if not os.path.exists(tar_path):
        print("Starting 1.7GB download. This may take a while depending on your internet connection...")
        download_file(url, tar_path)
    else:
        print(f"Found existing file at {tar_path}")
        
    # 2. Extract
    print("Extracting videos. Please wait...")
    with tarfile.open(tar_path, 'r') as tar:
        # Instead of generic extractall, we will ensure it goes directly into data/videos/
        for member in tqdm(tar.getmembers(), desc="Extracting"):
            if member.isfile() and member.name.endswith(".avi"):
                # member.name is like "YouTubeClips/something.avi"
                filename = os.path.basename(member.name)
                # target location
                dest_path = os.path.join("data", "videos", filename)
                
                # if already exists, skip
                if os.path.exists(dest_path):
                    continue
                    
                # write the file
                source = tar.extractfile(member)
                if source is not None:
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with open(dest_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                        
    print("All .avi videos have been successfully extracted to data/videos/ !")
    print("You can now delete data/YouTubeClips.tar if you wish to save disk space.")
