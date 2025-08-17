# Advanced CNN for CIFAR-10 Image Classification

## Problem Statement and Goal of Project

This project addresses the classic computer vision problem of image classification using the CIFAR-10 dataset. The primary goal was not just to build a functional classifier, but to architect a robust Convolutional Neural Network (CNN) from scratch and systematically enhance its performance, training efficiency, and reliability.

The focus was on implementing and evaluating advanced deep learning techniques, including custom learning rate schedulers, model calibration, and uncertainty estimation, to create a model that is both accurate and trustworthy.

## Solution Approach

The solution follows a comprehensive pipeline, starting from data preparation and moving through advanced model architecture design, training optimization, and detailed post-hoc analysis.

### 1\. Model Architecture

A custom CNN was designed as a `keras.models.Sequential` model with a focus on deep architecture and robust regularization to prevent overfitting.

  - **Convolutional Blocks:** The network consists of four sequential convolutional blocks with increasing filter depths (64 → 125 → 256 → 560) to capture features of increasing complexity.
  - **Regularization:** To ensure generalization, the model incorporates multiple regularization techniques:
      - **L2 Kernel Regularization** (`1e-4`)
      - **Kernel Constraint** (`max_norm`)
      - **Dropout** layers with varying rates (0.1 to 0.3) after convolutional and dense layers.
  - **Normalization:** `BatchNormalization` is used after every convolutional and dense layer to stabilize and accelerate training.
  - **Dense Head:** A `GlobalAveragePooling2D` layer reduces the feature maps before they are passed to two fully connected (`Dense`) layers, which act as the classifier head.
  - **Initialization:** `HeNormal` and `GlorotUniform` initializers are used for optimal weight initialization.

### 2\. Training Enhancements & Optimization

Several state-of-the-art techniques were employed to improve training dynamics and model performance.

  - **Mixed Precision Training:** Utilized TensorFlow's `mixed_float16` policy to leverage the tensor cores on my NVIDIA RTX 3070 Ti, significantly reducing training time and memory consumption without compromising accuracy.
  - **Data Augmentation:** An `ImageDataGenerator` was implemented to augment the training data in real-time. This included random rotations, shifts, and horizontal flips, which helps the model learn invariant features and generalize better.
  - **Advanced Learning Rate Scheduling:** I designed and implemented a custom `WarmUpCosineDecayReduceLROnPlateau` callback. This sophisticated scheduler:
    1.  **Warms up** the learning rate for the first 10% of steps to prevent initial instability.
    2.  Applies a **Cosine Decay** for smooth convergence towards the minimum.
    3.  Integrates a **Reduce on Plateau** mechanism to cut the learning rate if validation loss stagnates after the initial decay phase is complete.
  - **Callbacks:** Training was monitored using `EarlyStopping` to halt training when performance plateaus and `ModelCheckpoint` to save only the best-performing model based on validation loss.

### 3\. Model Reliability & Evaluation

Beyond standard accuracy metrics, I explored the model's reliability through calibration and uncertainty estimation.

  - **Uncertainty Quantification:** Implemented **Monte Carlo Dropout**, a Bayesian approximation technique, to estimate model uncertainty. By performing multiple forward passes with dropout enabled at inference time, I could calculate the standard deviation of predictions as a measure of the model's confidence.
  - **Confidence Calibration:** Used **Temperature Scaling**, a post-processing technique, to calibrate the model's output probabilities. An optimal temperature was found by minimizing the loss on the validation set, aligning the model's confidence scores more closely with its actual accuracy. This was visualized using a **Reliability Diagram**.

## Technologies & Libraries

  - **Frameworks:** TensorFlow, Keras
  - **Libraries:** NumPy, Scikit-learn, Matplotlib, Seaborn
  - **Tools:** Jupyter Notebook

## Description about Dataset

The project utilizes the **CIFAR-10 dataset**, a standard benchmark for image classification.

  - **Content:** 60,000 color images of size 32x32 pixels.
  - **Classes:** 10 distinct classes (e.g., airplane, automobile, bird, cat).
  - **Split:** The data was custom-split into training (72.25%), validation (12.75%), and test (15%) sets to ensure robust evaluation.

## Installation & Execution Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/imehranasgari/your-repo-name.git
    cd your-repo-name
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *Note: `requirements.txt` should contain:*
    ```
    tensorflow
    numpy
    scikit-learn
    matplotlib
    seaborn
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook project.ipynb
    ```
    Execute the cells sequentially to load data, build, train, and evaluate the model.

## Key Results / Performance

The final custom-built model demonstrated strong performance and well-calibrated confidence.

  - **Test Accuracy:** **85.28%**
  - **Test Loss:** 0.6719
  - **Best Validation Loss:** **0.6831** (achieved at epoch 29)
  - **Training Time:** 5024 seconds (\~84 minutes) on an NVIDIA GeForce RTX 3070 Ti Laptop GPU.
  - **Model Calibration:** The Reliability Diagram shows that Temperature Scaling successfully improved the model's calibration, making its confidence scores more indicative of its true accuracy.

## Sample Out

The following plots are generated by the notebook and illustrate the model's training progression and final performance.

#### Training History (Accuracy & Loss)

#### Learning Rate Schedule

#### Confusion Matrix & Reliability Diagram

## Additional Learnings / Reflections

This project was a deep dive into the practical aspects of building and optimizing a production-ready CNN.

  - **Custom Callbacks:** Implementing the `WarmUpCosineDecayReduceLROnPlateau` scheduler was a valuable exercise in controlling training dynamics at a granular level. It highlighted the significant impact that a well-designed learning rate schedule has on achieving stable and optimal convergence.
  - **Model Reliability:** Exploring MC Dropout and Temperature Scaling underscored the importance of not just accuracy, but also model calibration and uncertainty awareness. A model that "knows when it doesn't know" is far more useful in real-world applications.
  - **Exploratory Models:** While the main focus was the custom CNN, I also experimented with a pre-trained **ResNet50** model. This exploration, though not fully trained to completion in the notebook, demonstrated my understanding of transfer learning principles and different fine-tuning strategies (feature extraction vs. partial unfreezing).

-----

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

## 👤 Author

## Mehran Asgari

  - **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)
  - **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

-----

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.