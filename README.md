# deeplearningtask2

🚀 Overview
This project implements and trains three distinct types of neural networks using the FashionMNIST dataset and synthetic sequence data. It is designed to demonstrate:

Image Classification (CNN)

Sequential Data Processing (RNN, LSTM, GRU)

Generative Modeling (GAN)

🛠️ Architecture Details
1. Simple CNN
A standard architecture for image recognition tasks.

Layers: Two convolutional blocks (Conv2d + ReLU + MaxPool2d) followed by a fully connected classifier with Dropout.

Dataset: FashionMNIST (subset of 10,000 images for speed).

Output: Multi-class classification (10 categories of clothing).

2. RNN Comparison
A comparative study of different recurrent architectures for sequence handling.

Variants: Standard RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU).

Input: Dummy sequence data (integer sequences embedded into 32-dimensional vectors).

Output: Binary classification using a Sigmoid activation.


3. Generative Adversarial Network (GAN)A minimax game between two networks to generate realistic-looking images.Generator:
Transforms 100-dimensional noise into $28 \times 28$ grayscale images.Discriminator:
 A binary classifier that distinguishes between real dataset images and fake generated images.Training:
 Uses Binary Cross Entropy (BCE) loss to update both networks iteratively.


📊 Training & VisualizationThe script includes built-in visualization using matplotlib:CNN:
Plots the training loss curve and prints a Confusion Matrix to evaluate per-class performance.RNN:
Overlays the loss curves for RNN, LSTM, and GRU on a single plot to compare convergence speeds.GAN: 
Displays a $4 \times 4$ grid of generated images after every epoch and plots the adversarial loss (Generator vs. Discriminator).

Gemini said
This README provides a comprehensive overview of the cnn-rnn-gan.py script, which serves as a PyTorch-based educational suite for three fundamental deep learning architectures: Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Generative Adversarial Networks (GAN).

🚀 Overview
This project implements and trains three distinct types of neural networks using the FashionMNIST dataset and synthetic sequence data. It is designed to demonstrate:

Image Classification (CNN)

Sequential Data Processing (RNN, LSTM, GRU)

Generative Modeling (GAN)

🛠️ Architecture Details
1. Simple CNN
A standard architecture for image recognition tasks.

Layers: Two convolutional blocks (Conv2d + ReLU + MaxPool2d) followed by a fully connected classifier with Dropout.

Dataset: FashionMNIST (subset of 10,000 images for speed).

Output: Multi-class classification (10 categories of clothing).

2. RNN Comparison
A comparative study of different recurrent architectures for sequence handling.

Variants: Standard RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU).

Input: Dummy sequence data (integer sequences embedded into 32-dimensional vectors).

Output: Binary classification using a Sigmoid activation.

3. Generative Adversarial Network (GAN)
A minimax game between two networks to generate realistic-looking images.

Generator: Transforms 100-dimensional noise into 28×28 grayscale images.

Discriminator: A binary classifier that distinguishes between real dataset images and fake generated images.

Training: Uses Binary Cross Entropy (BCE) loss to update both networks iteratively.

📊 Training & Visualization
The script includes built-in visualization using matplotlib:

CNN: Plots the training loss curve and prints a Confusion Matrix to evaluate per-class performance.

RNN: Overlays the loss curves for RNN, LSTM, and GRU on a single plot to compare convergence speeds.

GAN: Displays a 4×4 grid of generated images after every epoch and plots the adversarial loss (Generator vs. Discriminator).

⚙️ Requirements
To run this script, you need the following Python libraries:

torch & torchvision

scikit-learn (for the confusion matrix)

matplotlib (for plotting)

numpy

