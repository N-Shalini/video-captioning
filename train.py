import os
import json
import glob
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_dataloader
from model import VideoCaptionModel
from config import Config

def train_model():
    print("Loading vocab...")
    with open('data/vocab.json', 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    print(f"Vocab size: {vocab_size}")
    dataloader = get_dataloader(vocab_path='data/vocab.json')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VideoCaptionModel(vocab_size, Config.FEATURE_DIM, Config.EMBED_DIM, Config.HIDDEN_DIM)
    
    start_epoch = 0
    # Check for existing checkpoints to resume training
    checkpoints = glob.glob(os.path.join(Config.MODELS_DIR, "caption_model_ep*.pth"))
    if checkpoints:
        epochs = []
        for cp in checkpoints:
            match = re.search(r'ep(\d+)\.pth', cp)
            if match:
                epochs.append(int(match.group(1)))
        if epochs:
            start_epoch = max(epochs)
            latest_checkpoint = os.path.join(Config.MODELS_DIR, f"caption_model_ep{start_epoch}.pth")
            print(f"Resuming training from epoch {start_epoch} (checkpoint: {latest_checkpoint})...")
            model.load_state_dict(torch.load(latest_checkpoint, map_location='cpu'))

    model.to(device)
    
    # Ignore <pad>
    pad_idx = vocab[Config.PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    print(f"Starting training on device: {device}")
    
    model.train()
    
    target_epochs = start_epoch + Config.EPOCHS
    for epoch in range(start_epoch, target_epochs):
        total_loss = 0.0
        
        # tqdm for batch info
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for i, (features, captions) in enumerate(progress_bar):
            features = features.to(device)
            captions = captions.to(device)
            
            # Use teacher forcing: input is captions[:, :-1], target is captions[:, 1:]
            input_caps = captions[:, :-1]
            target_caps = captions[:, 1:]
            
            optimizer.zero_grad()
            
            outputs = model(features, input_caps) # [batch_size, max_length-1, vocab_size]
            
            # Flatten outputs and targets
            outputs = outputs.reshape(-1, vocab_size)
            targets = target_caps.reshape(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Average Loss: {avg_loss:.4f}")
        
        # save every epoch (or best loss)
        torch.save(model.state_dict(), os.path.join(Config.MODELS_DIR, f"caption_model_ep{epoch+1}.pth"))
        
    print("Training finished.")

if __name__ == '__main__':
    train_model()
