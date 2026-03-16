import os
import json
import torch
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from config import Config
from model import VideoCaptionModel
from dataset import VideoCaptionDataset

# Ensure NLTK punkt is downloaded for tokenization if needed later
# nltk.download('punkt')

def plot_loss():
    history_path = os.path.join(Config.MODELS_DIR, "loss_history.json")
    if not os.path.exists(history_path):
        print(f"Could not find loss history at {history_path}. Please train the model first.")
        return
        
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss', marker='o', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    
    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(epochs)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(Config.MODELS_DIR, 'loss_curve.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nLoss curve successfully saved to '{plot_path}'")

def evaluate_bleu():
    print("\nLoading validation dataset for evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # We only need the validation split to compute BLEU
    val_dataset = VideoCaptionDataset("data/processed_captions.json", "data/vocab.json", Config.FEATURES_DIR, split="val")
    
    with open("data/vocab.json", 'r') as f:
        vocab = json.load(f)
        
    idx2word = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    model = VideoCaptionModel(vocab_size, Config.FEATURE_DIM, Config.EMBED_DIM, Config.HIDDEN_DIM)
    
    # Load the latest checkpoint
    import glob
    import re
    checkpoints = glob.glob(os.path.join(Config.MODELS_DIR, "attention_model_ep*.pth"))
    if not checkpoints:
        print("No trained models found! Please run train.py first.")
        return
        
    epochs = [int(re.search(r'ep(\d+)\.pth', cp).group(1)) for cp in checkpoints if re.search(r'ep(\d+)\.pth', cp)]
    latest_epoch = max(epochs) if epochs else Config.EPOCHS
    checkpoint_path = os.path.join(Config.MODELS_DIR, f"attention_model_ep{latest_epoch}.pth")
    
    print(f"Loading model checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Pre-group ground truth validation captions by video_id
    print("Grouping ground-truth captions...")
    vid_to_captions = {}
    with open("data/processed_captions.json", 'r') as f:
        all_caps = json.load(f)
        for c in all_caps:
            vid = c['video_id']
            # strip special tokens for BLEU evaluation
            tokens = [t for t in c['tokens'] if t not in [Config.BOS_TOKEN, Config.EOS_TOKEN, Config.PAD_TOKEN]]
            if vid not in vid_to_captions:
                vid_to_captions[vid] = []
            vid_to_captions[vid].append(tokens)

    # Get the unique videos in the validation set
    val_videos = list(set([item['video_id'] for item in val_dataset.data]))
    
    references = []
    hypotheses = []
    
    print(f"Generating captions for {len(val_videos)} validation videos...")
    with torch.no_grad():
        for vid_id in tqdm(val_videos):
            feat_path = os.path.join(Config.FEATURES_DIR, f"{vid_id}.npy")
            if not os.path.exists(feat_path):
                continue
                
            import numpy as np
            features = torch.FloatTensor(np.load(feat_path)).unsqueeze(0).to(device) # [1, MAX_FRAMES, FEATURE_DIM]
            
            # Generate feature using Encoder
            encoder_out, global_context = model.encoder(features)
            
            h = global_context
            c = torch.zeros_like(h)
            
            curr_token = torch.tensor([[vocab[Config.BOS_TOKEN]]]).to(device)
            caption_words = []
            
            for _ in range(Config.MAX_LENGTH):
                emb = model.decoder.embed(curr_token).squeeze(1)
                context_vector, _ = model.decoder.attention(encoder_out, h)
                
                lstm_input = torch.cat([emb, context_vector], dim=1)
                h, c = model.decoder.lstm_cell(lstm_input, (h, c))
                outputs = model.decoder.linear(h)
                
                pred_token = outputs.argmax(dim=-1).item()
                word = idx2word.get(pred_token, Config.UNK_TOKEN)
                
                if word == Config.EOS_TOKEN:
                    break
                    
                caption_words.append(word)
                curr_token = torch.tensor([[pred_token]]).to(device)
                
            hypotheses.append(caption_words)
            references.append(vid_to_captions[vid_id])

    print("\nCalculating BLEU scores...")
    # Corpus BLEU calculates the score across the entire dataset
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    print(f"==============================")
    print(f"BLEU-1 Score: {bleu1*100:.2f}")
    print(f"BLEU-2 Score: {bleu2*100:.2f}")
    print(f"BLEU-3 Score: {bleu3*100:.2f}")
    print(f"BLEU-4 Score: {bleu4*100:.2f}")
    print(f"==============================")

if __name__ == '__main__':
    plot_loss()
    evaluate_bleu()
