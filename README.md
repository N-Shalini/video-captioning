# MSVD Video Captioning Model built with VGG16 and LSTM

This project is an end-to-end framework to process, train, and test a deep learning Video Captioning Model using a Sequence-to-Sequence architecture (Encoder-Decoder) where spatial video features are extracted with VGG16 and captions are generated with a recurrent LSTM.

## Requirements

Ensure you have your environment set up. You can install all minimum dependencies via:
```sh
pip install -r requirements.txt
```

## Structure Organization

- `data/` : Folder to hold `video_corpus.csv` and the downloaded MSVD Videos.
- `data/videos/` : Store all your `.mp4` or `.avi` files here.
- `models/` : Stores checkpoints of your PyTorch model after each training epoch.
- `features/` : Preprocessed numpy array `.npy` vectors describing all video frames after VGG16 pooling are saved here.

## Step-by-Step Guideline: Training from Scratch on MSVD Data

### Step 1: Downloading & Prepping Data
Download the MSVD data:
1. Obtain the `video_corpus.csv` file from MSVD and stick it into `./data/`. It should have at minimum `VideoID` and `Description` columns.
2. Put the `.avi` video files into `./data/videos/`.
*(If the folder `data/` is missing or empty, scripts will create it and auto-generate mock placeholder data to allow validation testing without crashing.)*

### Step 2: Extracting VGG16 Frame Features
Machine learning over pure videos is extremely GPU expensive. Thus, we pre-process our video pixels by forwarding them through PyTorch's VGG16 to map frames down to 4096-dimension arrays. 

Run:
```sh
python extract_features.py
```
*Depending on how many videos you place in `data/videos/`, this could take a long time even on GPU! Please be patient.*

### Step 3: Organize Caption Vocabularies
Build standard JSON vocabulary mapping to encode tokenized English textual labels for our LSTM Sequence target.

Run:
```sh
python prep_data.py
```

### Step 4: Training The Encoder-Decoder Logic
Once we have our VGG extracted arrays and vocab logic, our neural network begins training using the Sequence-to-Sequence interface logic found within `model.py` and PyTorch DataLoader inside `dataset.py`.

To begin the supervised training routine (with progress bars and auto-checkpointing):
```sh
python train.py
```
*You can configure model constants directly in `config.py` like `EPOCHS = 10` or change embedding capacity configurations! Checkpoints natively save as `models/caption_model_ep*.pth`.*

### Step 5: Test & Infer on new files
Test an arbitrary video without needing to use MSVD dataset constraints by hooking inference script which sequentially extracts and predicts directly:
```sh
python inference.py --video path/to/my_random_video.mp4 --model models/caption_model_ep10.pth
```
