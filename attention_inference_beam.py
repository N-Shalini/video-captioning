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
import torch.nn.functional as F


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
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        frames.append(transform(img))
        
    cap.release()
    
    while len(frames) < Config.MAX_FRAMES:
        frames.append(torch.zeros(3, 224, 224))
        
    frames = frames[:Config.MAX_FRAMES]
    frame_batch = torch.stack(frames).to(device)
    
    with torch.no_grad():
        features = vgg(frame_batch)  # [MAX_FRAMES, 4096]
        features = features.unsqueeze(0)  # [1, MAX_FRAMES, 4096]
        
    # 3. Load Caption Model
    print("Loading caption generation model...")
    model = VideoCaptionModel(vocab_size, Config.FEATURE_DIM, Config.EMBED_DIM, Config.HIDDEN_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 4. Beam Search Caption Generation
    encoder_out = model.encoder(features)  # [1, max_frames, hidden_dim]
    
    beam_size = 3

    h = encoder_out.mean(dim=1)
    c = torch.zeros_like(h)

    # Beam: (sequence, score, h, c)
    beam = [([vocab[Config.BOS_TOKEN]], 0.0, h, c)]
    completed_sequences = []

    for _ in range(Config.MAX_LENGTH):
        new_beam = []

        for seq, score, h, c in beam:
            curr_token = torch.tensor([[seq[-1]]]).to(device)

            # If EOS → move to completed
            if seq[-1] == vocab[Config.EOS_TOKEN]:
                completed_sequences.append((seq, score))
                continue

            emb = model.decoder.embed(curr_token).squeeze(1)
            context_vector, _ = model.decoder.attention(encoder_out, h)

            lstm_input = torch.cat([emb, context_vector], dim=1)
            h_new, c_new = model.decoder.lstm_cell(lstm_input, (h, c))

            outputs = model.decoder.linear(h_new)
            log_probs = F.log_softmax(outputs, dim=-1)

            top_log_probs, top_indices = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_token = top_indices[0][i].item()
                next_score = score + top_log_probs[0][i].item()

                new_seq = seq + [next_token]
                new_beam.append((new_seq, next_score, h_new, c_new))

        # Keep top-k sequences
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]

        # Stop if all sequences ended
        if all(seq[-1] == vocab[Config.EOS_TOKEN] for seq, _, _, _ in beam):
            break

    # Add remaining sequences
    completed_sequences.extend([(seq, score) for seq, score, _, _ in beam])

    # Length normalization
    def normalize(seq, score):
        return score / len(seq)

    best_seq = max(completed_sequences, key=lambda x: normalize(x[0], x[1]))[0]

    # Convert tokens → words
    caption_words = []
    for token in best_seq:
        word = idx2word.get(token, Config.UNK_TOKEN)
        if word in [Config.BOS_TOKEN, Config.EOS_TOKEN]:
            continue
        caption_words.append(word)

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

    wrapped = textwrap.fill(caption, width=60)
    caption_lines = wrapped.split('\n')

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.75
    font_color = (255, 255, 255)
    font_thick = 1
    bar_height = 40 * len(caption_lines) + 20

    print(f"\nCaption: {caption}")
    print("Playing video... Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        for i, line in enumerate(caption_lines):
            text_size = cv2.getTextSize(line, font, font_scale, font_thick)[0]
            x = (w - text_size[0]) // 2
            y = h - bar_height + 30 + i * 40

            cv2.putText(frame, line, (x+1, y+1), font, font_scale, (0,0,0), font_thick+1, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), font, font_scale, font_color, font_thick, cv2.LINE_AA)

        cv2.imshow(window_title, frame)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
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