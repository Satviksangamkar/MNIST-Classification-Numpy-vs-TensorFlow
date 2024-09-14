# 🧠 MNIST Digit Classification: Numpy vs TensorFlow

Welcome to the MNIST Digit Classification project! This repository showcases different approaches for digit classification on the MNIST dataset, including a basic Numpy neural network, an advanced TensorFlow deep neural network, and a powerful Keras Convolutional Neural Network (CNN).

## 📂 Project Structure

- **`MNIST_Classification_Numpy_NeuralNetwork.ipynb`**: Explore a fundamental neural network implementation built from scratch using Numpy.
- **`MNIST_Classification_TensorFlow_DeepNN.ipynb`**: Dive into a sophisticated deep neural network built with TensorFlow.
- **`MNIST_Classification_TensorFlow_Keras_CNN.ipynb`**: Discover a high-performance Convolutional Neural Network (CNN) using Keras.
- **`mnist_train.csv`** and **`mnist_test.csv`**: The MNIST dataset for training and testing.

## 🚀 Getting Started

Follow these steps to set up your environment and start experimenting with the code:

### 1. Clone the Repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/Satviksangamkar/MNIST-Classification-Numpy-vs-TensorFlow.git
cd MNIST-Classification-Numpy-vs-TensorFlow
```

### 2. Set Up a Virtual Environment

Create and activate a virtual environment to manage dependencies:

```sh
python -m venv env
source env/bin/activate # On Windows use env\Scripts\activate
```

### 3. Install Required Packages

Install the necessary packages using `pip`:

```sh
pip install pandas numpy pillow matplotlib scikit-learn tensorflow keras jupyterlab
```

### 4. Download the MNIST Dataset

Obtain the MNIST dataset CSV files from Kaggle and place them in the root directory of the repository.

### 5. Run the Notebooks

Launch Jupyter Notebook and open the notebooks:

```sh
jupyter lab # Or jupyter notebook for the classic interface
```

Open each notebook (`MNIST_Classification_Numpy_NeuralNetwork.ipynb`, `MNIST_Classification_TensorFlow_DeepNN.ipynb`, `MNIST_Classification_TensorFlow_Keras_CNN.ipynb`) to run the code and explore the implementations.

## 🧩 Techniques & Methods

### 1. **Numpy Neural Network** 🛠️

A basic neural network implementation using Numpy:
- **Forward Propagation**: Computes outputs based on weights.
- **Backward Propagation**: Updates weights using gradient descent.
- **Activation Functions**: Includes Sigmoid and Softmax for classification.

### 2. **TensorFlow Deep Neural Network** 🔬

An advanced neural network utilizing TensorFlow:
- **Layer Abstraction**: Build models with dense layers and various activation functions.
- **Optimizers**: Employ advanced optimizers like Adam for efficient training.
- **Loss Functions**: Calculate cross-entropy loss for accurate classification.

### 3. **Keras Convolutional Neural Network (CNN)** 🌟

A Convolutional Neural Network using Keras for improved performance:
- **Convolutional Layers**: Extract image features with filters.
- **Pooling Layers**: Reduce dimensionality while retaining important features.
- **Regularization**: Implement dropout to prevent overfitting.

## 📈 Usage

- **Numpy Implementation**: Ideal for understanding the fundamentals of neural networks and their training.
- **TensorFlow Implementation**: Perfect for building and training more complex models.
- **Keras CNN**: Use for high-accuracy image classification tasks.

## 📄 License

This project is licensed under the MIT License. Feel free to contribute, modify, or use the code as per the license terms.

## 🔗 Links

- [MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)

## 📸 Example Predictions

You can visualize predictions and test the models with your own images. Here's a sample of how to use the trained models to predict digits from images.

**Image Example:**

```python
import matplotlib.pyplot as plt
from PIL import Image

def display_image(image_path):
    image = Image.open(image_path).convert('L')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

image_path = 'sample_image-300x298.webp'
display_image(image_path)
```
