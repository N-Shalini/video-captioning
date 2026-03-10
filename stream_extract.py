import urllib.request
import tarfile
import os

url = "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"
os.makedirs("data/videos", exist_ok=True)
print("Streaming tar...")
req = urllib.request.urlopen(url)
try:
    with tarfile.open(fileobj=req, mode="r|") as tar:
        extracted = 0
        for member in tar:
            if member.isfile() and member.name.endswith(".avi"):
                filename = os.path.basename(member.name)
                dest_path = os.path.join("data", "videos", filename)
                if not os.path.exists(dest_path):
                    f = tar.extractfile(member)
                    if f:
                        with open(dest_path, "wb") as out:
                            while True:
                                chunk = f.read(8192)
                                if not chunk: break
                                out.write(chunk)
                extracted += 1
                if extracted % 10 == 0:
                    print(f"Extracted {extracted} videos...")
except Exception as e:
    print(f"Stopped stream: {e}")
print("Done streaming.")
