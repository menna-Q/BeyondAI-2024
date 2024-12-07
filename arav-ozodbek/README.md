![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2024.png)

# Comprehensive Investigation of Double Descent Behavior with Dropout and Weight Decay Regularization

## Project Description

This research explores the double descent phenomenon, which challenges traditional notions of the bias-variance tradeoff in machine learning. Double descent demonstrates a non-monotonic relationship between model complexity and generalization performance. We investigate how regularization techniques—such as dropout and weight decay—affect double descent in mitigating overfitting across various model architectures.

### Research Question:
How do regularization techniques (such as dropout and weight decay) influence the double descent phenomenon and mitigate overfitting across different deep learning model architectures?

### Method and Implementation:
1. **Dataset:** Synthetic datasets were generated for regression and binary classification tasks. The regression dataset follows the formula:
    \[
    y = \sin(2 \pi x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
    \]
    with noise levels of 0.3 and 0.7. Binary classification was created by thresholding the regression outputs.

2. **Models:**
   - **Shallow Neural Networks:** Single hidden-layer networks were used to explore dropout effects.
   - **Decision Trees:** Depths were varied to analyze model-wise behaviors.
   - **Polynomial Regression:** Used for a controlled environment to observe double descent.

3. **Regularization Techniques:**
   - **Dropout:** Applied with rates from 0.0 to 0.4.
   - **Weight Decay:** L2 regularization with alpha values ranging from 0.01 to 1.0.

4. **Implementation:**
   - Python frameworks like TensorFlow/Keras and Scikit-learn were used for model training and evaluation.
   - Experiments were conducted using local hardware with VS Code for development.

### Results and Findings:
- Double descent was observed prominently in shallow networks and polynomial models.
- Dropout (0.1–0.2) effectively reduced overfitting but introduced instability at higher rates.
- Weight decay consistently smoothed loss curves and improved generalization.
- Combining dropout and weight decay provided the most consistent performance improvements.

### Conclusion:
This study highlights the critical role of regularization in managing overfitting and navigating the double descent landscape. By tailoring regularization techniques to specific architectures, practitioners can enhance model generalization and stability.

> **Find the research poster for this project in the [BeyondAI Proceedings 2024](https://thinkingbeyond.education/beyondai_proceedings_2024/).**

