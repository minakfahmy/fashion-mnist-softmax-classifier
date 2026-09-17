# Fashion-MNIST Softmax Regression Classifier

A clean, modular implementation of a Softmax Regression (Multinomial Logistic Regression) classifier built from scratch using NumPy and evaluated on the Fashion-MNIST dataset. This project includes mathematical derivations of gradients, vectorized implementation of forward and backward passes, L2 regularization, and automated hyperparameter optimization tracked via Weights & Biases (W&B).

## Features

- **Built from Scratch:** Core components—including Softmax computation, Cross-Entropy Loss, and analytical gradient calculation—are implemented purely in NumPy.
- **Numerical Stability:** Softmax implementation utilizes row-wise max subtraction to prevent numerical overflow.
- **Regularization:** Includes tunable L2 regularization within both loss computation and weight gradient updates.
- **Experiment Tracking:** Integrated with Weights & Biases (W&B) to monitor validation loss, accuracy, and training metrics across hyperparameter sweeps.
- **Visualization:** Includes utility functions to visualize learned weight matrices as image patches.

## Mathematical Formulation

1. **Softmax Function:**
   $$P(Y = c \mid \mathbf{x}) = \frac{e^{\mathbf{z}_c}}{\sum_{k} e^{\mathbf{z}_k}}, \quad \text{where } \mathbf{z} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

2. **Cross-Entropy Loss with $L_2$ Regularization:**
   $$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k}) + \frac{\lambda}{2} \Vert{}\mathbf{W}\Vert{}_F^2$$

3. **Gradients:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{1}{n} \mathbf{X}^T (\mathbf{\hat{Y}} - \mathbf{Y}) + \lambda \mathbf{W}$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \frac{1}{n} \sum_{i=1}^{n} (\mathbf{\hat{Y}} - \mathbf{Y})$$

## Project Structure

```text
├── dataset/
│   ├── fashion_mnist_train_images.npy
│   ├── fashion_mnist_train_labels.npy
│   ├── fashion_mnist_test_images.npy
│   └── fashion_mnist_test_labels.npy
├── main.py
├── requirements.txt
└── README.md
