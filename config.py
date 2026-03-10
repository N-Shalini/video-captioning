import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    FEATURES_DIR = os.path.join(BASE_DIR, 'features')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # MSVD Dataset files
    VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
    CAPTIONS_FILE = os.path.join(DATA_DIR, 'video_corpus.csv')
    
    # Model Hyperparameters
    MAX_FRAMES = 30           # Number of frames to sample per video
    FEATURE_DIM = 4096        # VGG16 fc2 layer output dimension
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    MAX_LENGTH = 30           # Maximum length of caption
    VOCAB_THRESHOLD = 2       # Minimum word count threshold
    
    # Training
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Preprocessing
    BOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    PAD_TOKEN = '<pad>'
    
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.FEATURES_DIR, exist_ok=True)
os.makedirs(Config.MODELS_DIR, exist_ok=True)
os.makedirs(Config.VIDEO_DIR, exist_ok=True)
