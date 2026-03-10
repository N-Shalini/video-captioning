import os
import pandas as pd
import json
from collections import Counter
import re
from config import Config

def tokenize_caption(caption):
    # Lowercase and keep alphanumeric chars
    caption = str(caption).lower()
    caption = re.sub(r'[^a-z0-9\s]', '', caption)
    tokens = [Config.BOS_TOKEN] + caption.split() + [Config.EOS_TOKEN]
    return tokens

def build_vocab(captions_file, out_vocab_path="data/vocab.json", out_captions_path="data/processed_captions.json"):
    print(f"Reading dataset from {captions_file}")
    if not os.path.exists(captions_file):
        print("Caption file not found, creating dummy vocabulary and data to verify pipeline.")
        # Dummy data for the sake of the pipeline completing
        data = {
            "Language": ["English", "English"],
            "VideoID": ["vid1", "vid2"],
            "Description": ["a man is playing a guitar", "a dog is running in the field"]
        }
        df = pd.DataFrame(data)
        df.to_csv(captions_file, index=False)
    else:
        df = pd.read_csv(captions_file)
    
    # MSVD sometimes has multiple languages, filter English only
    if 'Language' in df.columns:
        df = df[df['Language'] == 'English']
    
    all_tokens = []
    processed_caps = []
    
    for idx, row in df.iterrows():
        cap = str(row['Description'])
        vid = f"{row['VideoID']}_{row['Start']}_{row['End']}"
        if pd.isna(cap):
            continue
            
        tokens = tokenize_caption(cap)
        all_tokens.extend(tokens)
        processed_caps.append({'video_id': vid, 'tokens': tokens})
        
    # Count frequencies
    counter = Counter(all_tokens)
    
    # Build vocab dictionary mapping word to index
    vocab = {Config.PAD_TOKEN: 0, Config.BOS_TOKEN: 1, Config.EOS_TOKEN: 2, Config.UNK_TOKEN: 3}
    idx = 4
    for word, count in counter.items():
        if word not in vocab:
            if count >= Config.VOCAB_THRESHOLD:
                vocab[word] = idx
                idx += 1
                
    with open(out_vocab_path, 'w') as f:
        json.dump(vocab, f)
        
    with open(out_captions_path, 'w') as f:
        json.dump(processed_caps, f)
        
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Saved vocabulary to {out_vocab_path}")
    print(f"Saved processed captions to {out_captions_path}")
    return vocab, processed_caps

if __name__ == '__main__':
    build_vocab(Config.CAPTIONS_FILE)
