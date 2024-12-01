![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2024.png)

# A Comparative Analysis of Optimizers for Classification with CNN

## Project Description

This project investigates the impact of various optimization algorithms on the performance of Convolutional Neural Networks (CNNs) for image classification tasks. Selecting an appropriate optimizer is crucial for efficient training and robust performance of deep learning models.

### Motivation

Optimization algorithms play a pivotal role in determining the efficiency and accuracy of machine learning models. However, the question of which optimizer works best for specific tasks remains a critical challenge. By systematically comparing popular optimization algorithms, this research provides practical insights into their strengths and weaknesses in a controlled classification scenario.

### Research Question

*How do six commonly used optimization algorithms (SGD, Adam, AdaGrad, AdaDelta, RMSprop, and Nadam) compare in terms of accuracy, precision, recall, and F1-score for training a CNN on a binary image classification task?*

### Method and Implementation

- **Dataset:** Kaggle's "Dogs vs. Cats" dataset, containing 10,000 evenly distributed images.
- **CNN Architecture:**
  - Input layer: Resized images to 256×256 with three color channels.
  - Convolutional layers: Three layers with ReLU activation, max-pooling, and batch normalization.
  - Fully connected layers: Two dense layers (128 and 64 neurons) with ReLU activation, followed by a final output layer with sigmoid activation for binary classification.
- **Optimizers Evaluated:**
  - Stochastic Gradient Descent (SGD) with various learning rates.
  - Adam, AdaGrad, AdaDelta, RMSprop, and Nadam with default settings.
- **Metrics:** Accuracy, precision, recall, and F1-score were used to evaluate performance.
- **Frameworks:** TensorFlow and Keras libraries for implementation.

### Results

- **Best Overall Performance:** AdaGrad achieved the highest accuracy (79.09%) and F1-score (79.33%), demonstrating its effectiveness for this task.
- **High Precision:** Adam achieved the highest precision (84.23%), making it suitable for tasks emphasizing this metric.
- **High Recall:** Nadam performed best in recall (81.62%), excelling in detecting positive samples.
- **Batch Size Influence:** Batch size significantly influenced performance, with optimizers like RMSprop and Nadam performing better at higher batch sizes (512), while AdaGrad excelled at a batch size of 32.

### Conclusions

- AdaGrad emerged as the most balanced optimizer for this classification task, while Adam and Nadam showed strong performance in specific metrics.
- The study highlights the importance of selecting optimizers based on task requirements and dataset characteristics.
- While default settings often perform well for adaptive optimizers, hyperparameter tuning is crucial for algorithms like SGD.

> The research poster for this project can be found in the [BeyondAI Proceedings 2024](https://thinkingbeyond.education/beyondai_proceedings_2024/).
