# Digits Image Explorer

An interactive Streamlit application for exploring handwritten digit images through montages, intensity distributions, and occlusion saliency.

## Live App

https://digits-image-explorer-rana.streamlit.app/

## Dataset

The application uses the digits dataset included with scikit-learn:

- 1,797 grayscale images.
- Ten classes representing digits 0–9.
- Each image contains 8 × 8 pixels.
- Pixel intensities range from 0 to 16.

The dataset loads directly through `load_digits()`, so no separate download is required.

## Features

### Image Montage
Browse random image samples, filter by digit, and adjust the number of displayed images. The montage reveals variation in handwriting, including stroke thickness, slant, and shape. A bar chart summarizes class balance across the full dataset.

### Image Structure
Inspect an individual image alongside its pixel intensity histogram and view its underlying 8 × 8 matrix. Compare normalized intensity distributions between two digit classes.

### Model Explanation
Explore occlusion saliency for held-out test images using a logistic regression classifier. The model uses pixel values scaled to 0–1 and a stratified 80/20 train/test split with `random_state=42`.

The application masks overlapping patches with zeros and measures the change in probability for the original predicted class. Patch effects are averaged at each pixel:

- Red: masking reduced the predicted-class probability.
- Blue: masking increased the predicted-class probability.
- White: little change.

The original image, saliency map, and overlay are displayed together. Patch size is adjustable.

## Interpretation and Limitations

Images with the same label can differ in appearance, while different digits can share similar strokes. Intensity histograms summarize brightness but do not preserve spatial information.

Occlusion saliency explains the selected model's response to masking. Results depend on patch size and the masking value and should not be interpreted as an explanation of human digit recognition.

## Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Files

- `app.py`: Streamlit application.
- `requirements.txt`: Required Python libraries.
- `README.md`: Project overview and instructions.

## Tools

Python, Streamlit, NumPy, Matplotlib, and scikit-learn.
