# 🐱🐶 Cat vs Dog Image Classification

## 📌 Project Overview

This project builds a **deep learning model** to classify images of cats and dogs.
The dataset contains labeled images of cats and dogs, and the model learns to distinguish between the two classes using a **Convolutional Neural Network (CNN)**.

## 📂 Dataset

* **Source:** [Kaggle – Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
* **Classes:**

  * `cat` 🐱
  * `dog` 🐶
* **Images:** 25,000 training images (balanced dataset)

## ⚙️ Workflow

1. **Data Preprocessing**

   * Resize all images to fixed dimensions (e.g., 150x150).
   * Normalize pixel values (0–255 → 0–1).
   * Data augmentation (rotation, zoom, horizontal flip) to prevent overfitting.

2. **Model Architecture (CNN)**

   * Convolutional Layers (feature extraction).
   * MaxPooling Layers (downsampling).
   * Fully Connected Layers (classification).
   * Output Layer with **Sigmoid activation** (binary classification).

3. **Training**

   * Optimizer: Adam
   * Loss: Binary Crossentropy
   * Metrics: Accuracy

4. **Evaluation**

   * Accuracy and loss curves (training vs. validation).
   * Test set performance (accuracy, confusion matrix).

## 🛠️ Requirements

Install dependencies with:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```

## 🚀 Usage

1. Download the dataset and place it in a folder structure like:

   ```
   dataset/
   ├── train/
   │   ├── cats/
   │   └── dogs/
   └── validation/
       ├── cats/
       └── dogs/
   ```
2. Train the model:

   ```bash
   python train.py
   ```
3. Evaluate the model:

   ```bash
   python evaluate.py
   ```

## 📊 Example Results

* Training Accuracy: \~95%
* Validation Accuracy: \~90%
* Model successfully distinguishes cats from dogs with high accuracy.

## 📌 Future Improvements

* Use **transfer learning** with pre-trained models (VGG16, ResNet, MobileNet).
* Deploy model as a web app using **Flask/Streamlit**.
* Experiment with larger image sizes for better feature extraction.
