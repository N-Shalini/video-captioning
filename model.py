import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(feature_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.01)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x: [batch_size, max_frames, feature_dim]
        # Pool across frames:
        x = x.mean(dim=1) # [batch_size, feature_dim]
        x = self.linear(x) # [batch_size, hidden_dim]
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        # features: [batch_size, hidden_dim] -> this is context vector from Encoder
        # captions: [batch_size, max_length]
        
        # embed captions
        embeddings = self.embed(captions) # [batch_size, max_length, embed_dim]
        
        # typically we pass features first via sequence or as initial hidden state
        
        # Expand features to act as initial hidden & cell state
        # initial state requires [num_layers, batch_size, hidden_dim]
        h0 = features.unsqueeze(0) 
        c0 = torch.zeros_like(h0)
        
        # lstm forward pass
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        # [batch_size, max_length, hidden_dim]
        
        outputs = self.linear(self.dropout(lstm_out)) 
        # [batch_size, max_length, vocab_size]
        
        return outputs

class VideoCaptionModel(nn.Module):
    def __init__(self, vocab_size, feature_dim, embed_dim, hidden_dim):
        super(VideoCaptionModel, self).__init__()
        self.encoder = Encoder(feature_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim)
        
    def forward(self, video_features, captions):
        # Teacher forcing
        # Pass features through encoder
        context = self.encoder(video_features)
        
        # Decode
        outputs = self.decoder(context, captions)
        return outputs
