import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
from config import Config
from tqdm import tqdm

def extract_features(video_dir, out_dir, model, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    videos = [v for v in os.listdir(video_dir) if v.endswith(('mp4', 'avi'))]
    if len(videos) == 0:
        print(f"No videos found in {video_dir}. Creating mock videos to verify pipeline.")
        # create mock features to verify pipeline
        os.makedirs(out_dir, exist_ok=True)
        # Assuming the videos in prep_data.py were "vid1" and "vid2"
        mock_features = np.random.rand(Config.MAX_FRAMES, Config.FEATURE_DIM).astype(np.float32)
        np.save(os.path.join(out_dir, 'vid1.npy'), mock_features)
        np.save(os.path.join(out_dir, 'vid2.npy'), mock_features)
        return
        
    for vid in tqdm(videos):
        vid_path = os.path.join(video_dir, vid)
        vid_id = vid.split('.')[0]
        out_path = os.path.join(out_dir, f"{vid_id}.npy")
        
        if os.path.exists(out_path):
            continue
            
        cap = cv2.VideoCapture(vid_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # sample Config.MAX_FRAMES frames uniformly
        if frame_count > 0:
            indices = np.linspace(0, frame_count - 1, Config.MAX_FRAMES, dtype=int)
        else:
            indices = np.zeros(Config.MAX_FRAMES, dtype=int)
        
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = transform(img)
            frames.append(img)
            
        cap.release()
        
        # if not enough frames, pad with zeros matching the transformed image shape
        while len(frames) < Config.MAX_FRAMES:
            frames.append(torch.zeros(3, 224, 224))
            
        if len(frames) > Config.MAX_FRAMES:
            frames = frames[:Config.MAX_FRAMES]
            
        # Stack to batch [MAX_FRAMES, 3, 224, 224]
        frame_batch = torch.stack(frames).to(device)
        
        with torch.no_grad():
            features = model(frame_batch) # [MAX_FRAMES, FEATURE_DIM]
            
        features = features.cpu().numpy()
        np.save(out_path, features)
        
if __name__ == '__main__':
    # Define ResNet transformation pipeline natively compatible with VGG
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading VGG16 model...")
    # Using VGG16 to extract features from MSVD dataset
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # the classifier usually goes:
    # (0): Linear(in_features=25088, out_features=4096, bias=True)
    # (3): Linear(in_features=4096, out_features=4096, bias=True)
    # (6): Linear(in_features=4096, out_features=1000, bias=True)
    # We strip the last layer to output 4096-dim features
    vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1])
    
    print("Extracting features from videos...")
    os.makedirs(Config.FEATURES_DIR, exist_ok=True)
    extract_features(Config.VIDEO_DIR, Config.FEATURES_DIR, vgg, transform)
    print("Feature extraction complete.")
