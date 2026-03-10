import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from config import Config

class VideoCaptionDataset(Dataset):
    def __init__(self, captions_path, vocab_path, features_dir):
        with open(captions_path, 'r') as f:
            self.captions = json.load(f)
            
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
            
        self.features_dir = features_dir
        
        # valid data filtering: only keep captions where corresponding feature file exists
        self.data = []
        for c in self.captions:
            vid_id = c['video_id']
            feat_path = os.path.join(self.features_dir, f"{vid_id}.npy")
            if os.path.exists(feat_path):
                self.data.append(c)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid_id = item['video_id']
        tokens = item['tokens']
        
        # load video feature
        feat_path = os.path.join(self.features_dir, f"{vid_id}.npy")
        video_features = np.load(feat_path) # [MAX_FRAMES, FEATURE_DIM]
        video_features = torch.FloatTensor(video_features)
        
        # map tokens to ID array
        token_ids = []
        for word in tokens:
            token_ids.append(self.vocab.get(word, self.vocab[Config.UNK_TOKEN]))
            
        # truncate or pad sequence
        if len(token_ids) > Config.MAX_LENGTH:
            token_ids = token_ids[:Config.MAX_LENGTH]
        else:
            token_ids = token_ids + [self.vocab[Config.PAD_TOKEN]] * (Config.MAX_LENGTH - len(token_ids))
            
        caption = torch.LongTensor(token_ids)
        
        return video_features, caption

def get_dataloader(captions_path="data/processed_captions.json", 
                     vocab_path="data/vocab.json", 
                     features_dir=Config.FEATURES_DIR, 
                     batch_size=Config.BATCH_SIZE, 
                     shuffle=True):
    dataset = VideoCaptionDataset(captions_path, vocab_path, features_dir)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
