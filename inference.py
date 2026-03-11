import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from model import VideoCaptionModel
from config import Config
import argparse
import textwrap

def infer_video(video_path, model_path, vocab_path='data/vocab.json'):
    # Load vocab
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    idx2word = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Load VGG16
    print("Loading feature extractor VGG16...")
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1])
    vgg = vgg.to(device)
    vgg.eval()
    
    # 2. Extract features
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count > 0:
        indices = np.linspace(0, frame_count - 1, Config.MAX_FRAMES, dtype=int)
    else:
        indices = np.zeros(Config.MAX_FRAMES, dtype=int)
        
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        frames.append(transform(img))
        
    cap.release()
    
    while len(frames) < Config.MAX_FRAMES:
        frames.append(torch.zeros(3, 224, 224))
        
    frames = frames[:Config.MAX_FRAMES]
    frame_batch = torch.stack(frames).to(device)
    
    with torch.no_grad():
        features = vgg(frame_batch) # [MAX_FRAMES, 4096]
        features = features.unsqueeze(0) # [1, MAX_FRAMES, 4096]
        
    # 3. Load Caption Model
    print("Loading caption generation model...")
    model = VideoCaptionModel(vocab_size, Config.FEATURE_DIM, Config.EMBED_DIM, Config.HIDDEN_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 4. Generate caption greedily
    context = model.encoder(features) # [1, hidden_dim]
    
    h_c = (context.unsqueeze(0), torch.zeros_like(context.unsqueeze(0)))
    
    # start with BOS token
    curr_token = torch.tensor([[vocab[Config.BOS_TOKEN]]]).to(device) 
    
    caption_words = []
    
    for _ in range(Config.MAX_LENGTH):
        # embed current token
        emb = model.decoder.embed(curr_token) # [1, 1, embed_dim]
        lstm_out, h_c = model.decoder.lstm(emb, h_c)
        outputs = model.decoder.linear(lstm_out) # [1, 1, vocab_size]
        
        pred_token = outputs.argmax(dim=-1).item()
        word = idx2word.get(pred_token, Config.UNK_TOKEN)
        
        if word == Config.EOS_TOKEN:
            break
            
        caption_words.append(word)
        curr_token = torch.tensor([[pred_token]]).to(device)
        
    return " ".join(caption_words)

def play_video_with_caption(video_path, caption):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    window_title = "Video Captioning"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 800, 500)

    # Wrap caption if too long
    wrapped = textwrap.fill(caption, width=60)
    caption_lines = wrapped.split('\n')

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.75
    font_color = (255, 255, 255)
    font_thick = 1
    bar_height = 40 * len(caption_lines) + 20

    print(f"\nCaption: {caption}")
    print("Playing video... Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w = frame.shape[:2]

        # Draw semi-transparent black bar at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Draw each caption line
        for i, line in enumerate(caption_lines):
            text_size = cv2.getTextSize(line, font, font_scale, font_thick)[0]
            x = (w - text_size[0]) // 2
            y = h - bar_height + 30 + i * 40
            # Shadow for readability
            cv2.putText(frame, line, (x+1, y+1), font, font_scale, (0,0,0), font_thick+1, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y),   font, font_scale, font_color, font_thick, cv2.LINE_AA)

        cv2.imshow(window_title, frame)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help="Path to mp4 or avi video file")
    parser.add_argument('--model', type=str, default='models/caption_model_ep20.pth', help="Path to trained .pth model file")
    args = parser.parse_args()

    result = infer_video(args.video, args.model)
    print("--------------------------------------------------")
    print(f"Prediction for '{args.video}':\n\t{result}")
    print("--------------------------------------------------")

    play_video_with_caption(args.video, result)
