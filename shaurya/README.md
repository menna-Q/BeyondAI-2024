![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2024.png)

# DOUBLE DESCENT VS OVERFITTING IN DEEP LEARNING

1. The interplay between overfitting and double descent in deep learning challenges traditional learning theory. Investigating this transition reveals how model capacity, data properties, and regularization influence generalization. Understanding these dynamics is crucial for optimizing overparameterized models and addressing practical challenges in training large neural networks on diverse and noisy datasets.
   
2. Overfitting, a classical issue in deep learning, occurs when models perform well on training data but poorly on unseen data. Double descent challenges this notion, revealing that model performance can improve beyond the interpolation threshold as capacity increases, offering new insights into the dynamics of overparameterized models.

3. Model Design: A flexible feedforward neural network (FlexibleNN) was developed to study the effects of model complexity on performance.

A.Data Preparation: The MNIST dataset was used for training and evaluation, consisting of grayscale images of handwritten digits.

B.Training and Evaluation Pipeline: The model was trained using cross-entropy loss and SGD optimizer, with training and testing across epochs to analyze performance trends like overfitting and double descent.

C.Visualization: Training and test losses were plotted across epochs for each model configuration. This allowed the identification of overfitting regions and the double descent behavior as model complexity increased.

4. Effect of Model Complexity: 
A. Increasing the number and size of hidden layers initially improved performance, reducing both training and test loss.

B. Beyond a certain complexity, overfitting occurred, evidenced by a significant divergence between training and test loss.

C.Double Descent Phenomenon: For highly complex models, test loss showed an initial increase (overfitting) followed by a decrease as model capacity grew further, demonstrating double descent behavior.

Performance Trends:  
A. Simpler models struggled to capture patterns effectively, resulting in higher training and test loss.

B.Intermediate-complexity models balanced generalization and overfitting, achieving optimal performance.

C.Loss Visualization: Training and test loss curves provided clear visual evidence of overfitting and the transition to the double descent phase as model complexity increased.

5. This study demonstrated the double descent phenomenon, where increasing model complexity initially caused overfitting but later improved generalization. These findings highlight the complex relationship between model capacity and performance.

> The research poster for this project can be found in the [BeyondAI Proceedings 2024](https://thinkingbeyond.education/beyondai_proceedings_2024/).
