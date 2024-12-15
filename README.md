# Fashion MNIST Classification Using CNN and Spark

This project classifies clothing items into 10 categories using the Fashion MNIST dataset. It implements a Convolutional Neural Network (CNN) for image classification and leverages Apache Spark for distributed computing to optimize computational efficiency.

---

## 🧾 Project Overview

- **Goal**: Build a CNN to classify clothing items into categories such as T-shirt, Dress, Sneakers, etc.
- **Dataset**: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) (60,000 training images, 10,000 test images).
- **Tech Stack**: TensorFlow, Keras, PySpark, Scikit-learn, Matplotlib.

### Real-World Applications
- **E-commerce Platforms**: Automating product classification for online stores like Amazon, Zara, etc.
- **Visual Search**: Enable search for similar products by image.
- **Caption Automation**: Generate automated product descriptions based on image classification.

---

## 🛠️ Implementation Details

### Tools and Libraries
- **Data Handling**: Apache Spark
- **Model Training**: TensorFlow and Keras
- **Hyperparameter Tuning**: GridSearchCV
- **Visualization**: Matplotlib

### Model Architecture
- Convolutional layers for feature extraction.
- Dropout layers to prevent overfitting.
- Fully connected layers for classification.

### Hyperparameter Tuning
- **Learning rate**: {0.01, 0.001}
- **Dropout**: {0.3, 0.5}
- **Batch size**: {32, 64}

GridSearchCV with 3-fold cross-validation was used for tuning.

---

## 📊 Performance

- **Test Accuracy**: 90.86%
- **Test Loss**: 0.2506
- **Training Efficiency**:
  - Two VMs: 3.5 hours
  - Distributed over 4 VMs: 2.2 hours

---
