# MNIST Neural Network from Scratch 🚀

A lightweight, pure-Python implementation of a **2-layer Multilayer Perceptron (MLP)** built entirely from scratch using **NumPy**. This project classifies handwritten digits from the famous MNIST dataset without relying on deep learning frameworks like TensorFlow or PyTorch.

---

## 🧠 Network Architecture

* **Input Layer:** 784 neurons (representing 28x28 flattened pixel images).
* **Hidden Layer:** 64 neurons utilizing the **ReLU** activation function and He weight initialization.
* **Output Layer:** 10 neurons utilizing the **Softmax** activation function to output class probabilities for digits 0–9.
* **Training Algorithm:** Gradient Descent with manual backpropagation.

---

## 🛠️ Prerequisites & Dependencies

Make sure you have the required Python libraries installed before running the script:

```bash
pip install numpy pandas matplotlib scikit-learn
```

# 📊 Dataset Setup (train.csv)
This script expects a CSV file named train.csv in the root directory (formatted like the Kaggle Digit Recognizer dataset, where the first column is the label and the remaining 784 columns are pixel values).

You can set it up using one of two methods:

## Option 1: Download from Kaggle
Head over to the Kaggle Digit Recognizer Competition.

Download train.csv and place it in your project folder.

## Option 2: Generate Automatically via Python
Run this quick snippet to fetch MNIST via scikit-learn and format it into the expected CSV structure:

```python
import pandas as pd
from sklearn.datasets import fetch_openml

print("Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

df = pd.DataFrame(mnist.data)
df.insert(0, 'label', mnist.target.astype(int))

df.to_csv('train.csv', index=False)
print("train.csv created successfully!")
```
# 🚀 How to Run
1. Ensure train.csv is in the same directory as your script.
2. Execute the Python script:
```bash
python main_np.py
```
3. Watch the terminal for training progress updates (printing accuracy every 50 iterations).

# 🔍 Visualizing Predictions
To visually inspect predictions against the actual labels, uncomment the testing block at the bottom of the script:
```python
test_prediction(0)
test_prediction(1)
```

This will output the model's prediction, the true label, and pop up a matplotlib window showing the handwritten digit.
