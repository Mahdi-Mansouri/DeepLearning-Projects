# Deep Learning Course Notebooks

This repository contains my Jupyter Notebook implementations and assignments from the **Deep Learning** course that I eagerly audited. The materials cover fundamental and advanced topics such as neural networks, convolutional networks, transformers, generative models, and recent trends in self-supervised and foundation models.

Each session includes notebooks with code experiments, implementations of algorithms, and analyses.

---

## 📚 Course Topics

| Session | Topic |
|:---:|:---|
| 1 | Introduction to Deep Learning & Machine Learning Review |
| 2 | MLP I – Neural Networks, Universal Approximation Theorem |
| 3 | MLP II – Training Neural Networks & Back-Propagation |
| 4 | Optimization |
| 5 | Optimization & Generalization |
| 6 | Training Techniques |
| 7 | Convolutional Neural Networks (CNN) |
| 8 | CNN Architectures |
| 9 | CNNs in Vision Problems (Segmentation, Object Detection, etc.) |
| 10 | RNN I – RNNs and LSTMs |
| 11 | RNN II – Word Embeddings, Language Modeling |
| 12 | Attention & Transformer I |
| 13 | Transformer II – BERT, T5, GPT, ViT |
| 14 | Transformer III – Large Language Models |
| 15 | Generative Models I – Variational Autoencoders (VAE) |
| 16 | Generative Models II – Generative Adversarial Networks (GAN) |
| 17 | Generative Models III – Diffusion Models |
| 18 | Generative Models IV – Diffusion Models (Continued) |
| 19 | Self-supervised Learning I – Pretext Tasks, Contrastive Learning |
| 20 | Self-supervised Learning II – Losses and Architectures |
| 21 | Recent Foundation Models – CLIP, DALL·E |
| 22 | Graph Neural Networks |
| 23 | Visualization & Interpretability |
| 24 | Adversarial Robustness |
| 25 | Advanced Topics |

---

## 📝 Homeworks and Projects

Below you will find the links to each session’s notebooks with a brief description of their content.

---

### HW 1 – Foundations of Deep Learning

- [Notebook 1: Optimization Algorithms](HW1/HW1_PART1.ipynb)  
  Implementation of optimization algorithms including Gradient Descent, Momentum, RMSProp, Adam, and the Newton Method.
- [Notebook 2: Implementing an MLP from scratch](HW1/Neural_Networks_from_scratch_with_numpy.ipynb)  
  Development of a multi-layer perceptron entirely in NumPy, evaluated on the MNIST and California Housing datasets.
- [Notebook 3: PyTorch Tutorial](HW1/pytorch_basic.ipynb)  
  Introduction to PyTorch tensors and their practical challenges. Includes the implementation of a simple MLP using PyTorch for the MNIST dataset.
  
---

### **HW2: Convolutional Neural Network Implementations**

* **[Notebook 1: Image Classification and Colorization](HW2/CIFAR10_Classification_And_Colorization.ipynb)**
    * **ResNet-18:** An implementation of the ResNet-18 architecture for image classification, trained on the CIFAR-10 dataset.
    * **UNet:** A UNet model built and trained to perform automatic image colorization on grayscale inputs.
* **[Notebook 2: CNN from Scratch](HW2/HW2_CNN_TODO.ipynb)**
    * Demonstrates the fundamental building blocks of CNNs by implementing a standard network from scratch in PyTorch.
    * Features a custom data augmentation pipeline to improve model generalization.
* **[Notebook 3: YOLOv2 for Object Detection](HW2/HW2_YOLO_TODO.ipynb)**
    * An implementation of the YOLOv2 (You Only Look Once v2) architecture, a highly efficient model for real-time object detection.
---

### **HW3: RNN and LSTM and Transformer**

* **[Notebook 1: RNN and LSTM](HW3/RNN.ipynb)**
    * **RNN:** An implementation of a traditional RNN architecture to predict nationality based on the name.
    * **LSTM:** An implementation of LSTM architecture for sentiment analysis (on imdb dataset)
* **[Notebook 2: SimpleGPT](HW3/simple_GPT-NO-OUTPUT.ipynb)**
    * Implemented a Transformer block from scratch using pytorch.
    * Implemented a decoder-only Language Model to generate Friends dialogues.
* **[Notebook 3: Bert for Masked Language Modeling and Sequence Classification](HW3/Bert_MLM_SeqClassification-NO-OUTPUT.ipynb)**
    * Implemented and trained a bert for masked LM and sequence classification by training the bert by myself and hugging face trainer.
---
