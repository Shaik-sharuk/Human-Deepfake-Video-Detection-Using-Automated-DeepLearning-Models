# Human Deepfake Detection Using CNN and LSTM

A Final Year B.Tech project that detects whether a human face video is **real** or **deepfake** by analyzing video frames using:
- **CNN** for spatial feature extraction
- **LSTM** for temporal pattern learning

The system is designed to run on a normal laptop and is suitable for academic submission and demonstration.

---

## System Workflow

1. User uploads a video
2. Frames are extracted using OpenCV
3. Faces are detected and cropped (Haar Cascade)
4. Frames are resized to 64×64 and normalized
5. Frames are converted into sequences (10 frames per sequence)
6. CNN extracts spatial features from each frame
7. LSTM learns temporal relationships between frames
8. Model predicts **Real** or **Deepfake** with confidence score
9. Result is displayed on the web interface

---

## Project Structure

```
deepfake_detection_project/
├── dataset/                 # Place your real/fake video datasets here
│   ├── real/                # Real human face videos
│   └── fake/                # Deepfake videos
├── models/
│   └── cnn_lstm_model.py    # CNN + LSTM model architecture
├── preprocessing/
│   ├── extract_frames.py   # Extract frames from video
│   ├── face_detection.py   # Detect and crop faces
│   └── data_loader.py     # Load and preprocess data for training
├── training/
│   ├── train_model.py     # Training script
│   └── evaluate_model.py  # Evaluation, confusion matrix, graphs
├── webapp/
│   ├── app.py             # Flask application
│   ├── templates/
│   │   ├── index.html     # Upload page
│   │   └── result.html    # Prediction result page
│   └── static/
│       └── style.css      # Styling
├── utils/
│   └── video_utils.py     # Video processing utilities
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

- Create folders: `dataset/real/` and `dataset/fake/`
- Place **real** human face videos in `dataset/real/`
- Place **deepfake** videos in `dataset/fake/`
- Use short clips (e.g., 2–10 seconds) with visible faces for best results

**Sample datasets (academic use):**
- [FaceForensics++](https://github.com/ondyari/FaceForensics) (subset)
- [DFDC Preview](https://www.kaggle.com/c/deepfake-detection-challenge) (small subset)
- Or any labeled real/fake face video clips

---

## Training the Model

From the project root (`deepfake_detection_project/`):

```bash
python run_train.py
# Or with options:
python run_train.py --epochs 20 --batch_size 16 --max_videos 50
# Or using module:
python -m training.train_model
```

**Training configuration:**
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Epochs: 10–20 (configurable)
- Batch size: 16
- Input: sequences of 10 frames, each frame 64×64×3

Trained model is saved as `models/saved_model.h5`.

---

## Evaluation

After training, run evaluation to get accuracy, loss curve, and confusion matrix:

```bash
python run_evaluate.py
# Or: python -m training.evaluate_model
```

Outputs are saved in `training/outputs/` (accuracy/loss plots and confusion matrix).

---

## Running the Web Application

1. Ensure the model is trained and `models/saved_model.h5` exists.
2. Start the Flask server (from project root):

```bash
python run_webapp.py
# Or: python -m webapp.app
```

3. Open a browser and go to: **http://127.0.0.1:5000**
4. Upload a video and click **Detect Deepfake** to get the prediction and confidence.

---

## Model Architecture Summary

| Layer Type        | Details                                      |
|-------------------|----------------------------------------------|
| Input             | (10, 64, 64, 3) — 10 frames, 64×64 RGB       |
| Conv2D            | 32 filters, 3×3, ReLU                        |
| MaxPooling2D      | 2×2                                          |
| Conv2D            | 64 filters, 3×3, ReLU                        |
| MaxPooling2D      | 2×2                                          |
| Flatten           | —                                            |
| LSTM              | 64 units                                     |
| Dense             | 64 units, ReLU                               |
| Dense             | 1 unit, Sigmoid (Real/Fake)                  |

---

## Output Example

**Result page:**
- **Prediction:** Deepfake Detected / Real
- **Confidence:** e.g., 89%

---

## Tips for Academic Use

- Start with a small balanced dataset (e.g., 50–100 videos per class) for quick training.
- Increase epochs if validation loss is still decreasing.
- For demos, use short clips (3–5 s) with a single clear face.
- If you get out-of-memory errors, reduce batch size in `training/train_model.py`.

---

## License

This project is for academic and educational purposes only.
