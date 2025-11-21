# Hand Hunter v2 🖐️
Autoencoder-based hand detection model.

This project detects whether an image contains a human hand by measuring
reconstruction error from a trained autoencoder.

## Features
- 96x96 convolutional autoencoder
- TensorFlow 2.17
- FastAPI inference server
- Optional real-time webcam script
- Docker-ready deployment

## Running with Docker

Build:

```
docker build -t handhunter .
```

Run API:

```
docker run -p 8000:8000 handhunter
```

Then test:

```
curl -X POST -F "file=@tests/test_imgs/hand.jpg" http://localhost:8000/detect
```

## Local Inference

```
python src/inference.py --image tests/test_imgs/myimage.jpg
```

## Model Testing (auto-calibrated threshold)
Use the helper script to auto-calibrate a MAE threshold from normal images and test a target image (defaults to the image's folder if not provided):

```
python src/test_model.py --auto-calibrate --image tests/test_imgs/closeup-of-black-female-hand-isolated-on-white-background.jpg
```

Or calibrate from a specific folder and save the reconstruction:

```
python src/test_model.py --auto-calibrate --calibrate-dir tests/test_imgs --save tests/test_imgs/reconstruction.png
```

## Model
Trained on ~66,000 augmented hand images using MAE reconstruction loss.

Model file:
- `model/hand_hunter.keras`
