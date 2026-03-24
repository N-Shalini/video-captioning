import os
import json
import torch
import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from model import VideoCaptionModel
from config import Config

# Ensure nltk datasets are available silently
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def compute_metrics():
    print("Loading vocab...")
    with open('data/vocab.json', 'r') as f:
        vocab = json.load(f)
    idx2word = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading model...")
    model = VideoCaptionModel(vocab_size, Config.FEATURE_DIM, Config.EMBED_DIM, Config.HIDDEN_DIM)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load all ground truths
    with open('data/processed_captions.json', 'r') as f:
        all_captions = json.load(f)
        
    gt_dict = {}
    for cap in all_captions:
        vid = cap['video_id']
        words = [w for w in cap['tokens'] if w not in ['<sos>', '<eos>', '<pad>']]
        if vid not in gt_dict:
            gt_dict[vid] = []
        gt_dict[vid].append(words)
        
    # Get available validation features
    feature_files = [f for f in os.listdir(Config.FEATURES_DIR) if f.endswith('.npy')]
    valid_vids = [f.replace('.npy', '') for f in feature_files if f.replace('.npy', '') in gt_dict]
    
    # Select a subset of validation videos to evaluate (default 300 to keep it somewhat quick)
    import random
    random.seed(42)  # Strict seed so metrics are reproducible
    valid_vids = list(set(valid_vids))
    random.shuffle(valid_vids)
    split_idx = int(len(valid_vids) * 0.8) # 80% train, 20% val
    val_vids = valid_vids[split_idx:]
    test_vids = val_vids[:300] if len(val_vids) > 300 else val_vids
    
    print(f"Evaluating {len(test_vids)} validation videos for metrics...\n")
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    rouge_l_scores = []
    
    for vid_id in tqdm(test_vids, desc="Generating Captions"):
        feat_path = os.path.join(Config.FEATURES_DIR, f"{vid_id}.npy")
        features = np.load(feat_path)
        features = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        # Greedy Decode
        with torch.no_grad():
            encoder_out = model.encoder(features)
            h = encoder_out.mean(dim=1)
            c = torch.zeros_like(h)
            
            curr_token = torch.tensor([[vocab[Config.BOS_TOKEN]]]).to(device)
            pred_words = []
            
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
                pred_words.append(word)
                curr_token = torch.tensor([[pred_token]]).to(device)
                
        # Calculate metrics
        ground_truths = gt_dict[vid_id]  
        
        # BLEU scores
        b1 = sentence_bleu(ground_truths, pred_words, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        b2 = sentence_bleu(ground_truths, pred_words, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        b3 = sentence_bleu(ground_truths, pred_words, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        b4 = sentence_bleu(ground_truths, pred_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        bleu1_scores.append(b1)
        bleu2_scores.append(b2)
        bleu3_scores.append(b3)
        bleu4_scores.append(b4)
        
        # ROUGE-L score requires string input
        pred_str = " ".join(pred_words)
        best_rouge = 0
        for gt_words in ground_truths:
            gt_str = " ".join(gt_words)
            score = scorer.score(gt_str, pred_str)['rougeL'].fmeasure
            if score > best_rouge:
                best_rouge = score
        rouge_l_scores.append(best_rouge)
        
    print("\n" + "="*50)
    print("🧠 FINAL PERFORMANCE METRICS 🧠")
    print("="*50)
    print(f"BLEU-1:  {np.mean(bleu1_scores)*100:.2f}")
    print(f"BLEU-2:  {np.mean(bleu2_scores)*100:.2f}")
    print(f"BLEU-3:  {np.mean(bleu3_scores)*100:.2f}")
    print(f"BLEU-4:  {np.mean(bleu4_scores)*100:.2f}")
    print(f"ROUGE-L: {np.mean(rouge_l_scores)*100:.2f}")
    print("="*50)
    print("Note: Scores are percentages (0-100%). Higher is better.")
    print("Copy this table for your reviewers!")

if __name__ == '__main__':
    compute_metrics()
