![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2024.png)

# MNIST Digit Classification: Comparing Classical and Quantum Approaches to Hyperparameter Tuning

1. We wanted to explore different aspects of ML models and focus on a specific topic. After discussing within the team and with our mentor, Dr. Emilie Gregoire, we came to the idea to focus on hyperparameter tuning. 
2. Research Question: Compare the optimal performance that Support Vector Machines (SVMs) and Multi-Layer Perceptrons (MLPs) can achieve using hyperparameter tuning. 
3. Method and Implementation: We used the classical method for the SVMs, looping over both the regularization constant, the C parameter and the gamma parameter or the kernel modifier of the Radial Basis Function (RBF) kernel used in the SVM model. The MLP hyperparameters were tuned based on a quantum approach using quantum circuits implemented using rotation gates.
4. Results:
- **For SVM Model:**
  - Using fixed hyperparameters:
    - Gamma = 0.05
    - C = 1.0
    - Accuracy: **0.9755**
  - Variation of Learning Time and Accuracy:
    - **With respect to Gamma:**
      - Accuracy peaked at **Gamma = 0.026**
      - Reached a peak accuracy of **0.967**
      - Regularization Constant (C) set as **1**
    - **With respect to C:**
      - Accuracy peaked at **C = 1.26 to 1.3**
      - Reached a peak accuracy of **0.963** at **Gamma = 0.026** (Peak Accuracy in Hyperparameter Tuning)
6. Draw your conclusions
- **For the SVM Model:**
  - A combination of:
    - **Gamma = 0.05** and **C = 1.0**
    - Returned the highest accuracy of **0.9755**.
  - **Gamma Parameter:**
    - Accuracy rate:
      - Increased with the value of Gamma.
      - Plateaued and then decreased with roughly the same slope.
      - Peaked at **Gamma = 0.926** with a peak accuracy of **0.967**.
    - Learning time:
      - Initially dropped with the value of Gamma.
      - Increased thereafter and plateaued towards the end.
  - **C Parameter (Regularization Constant):**
    - Accuracy rate:
      - Initially increased very sharply with the value of C.
      - Plateaued towards the end.
      - Peaked at **C = 1.26 to 1.3** with a peak accuracy of **0.963**.
    - Learning time:
      - Initially decreased very sharply with the value of C.
      - Plateaued towards the end.

> The research poster for this project can be found in the [BeyondAI Proceedings 2024](https://thinkingbeyond.education/beyondai_proceedings_2024/).
