![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2024.png)

# Limitations of Multilayer Perceptrons


The code shows different versions of the MLP arranged from most efficient to least efficient (project setbacks), each has its epoch evaluation printed as a quantitative aspect of comparison, and a method of monitoring the training process and checking for possible errors.
Quantitative aspects of the evaluation process mainly depended on the loss values of epochs and their plotted graphs, alternating the code and experimenting different numbers and activations gave a slight change in the loss values for each model, however, given the simplicity of these models, if the same methods were implemented in more complex MLPs the losses would increase exponentially. 
Another crucial quantitative aspect was the time of execution for each model, having computationally expensive MLPs can dramatically affect the efficiency of the tasks done thus making them an unpreferred solution for real-life implementations.

Qualitative aspects are represented in the choices for activation functions, initializations, and number of epochs, each of these is a factor that can affect the behavior of the model, however, the way to create the best MLP possible is through trying to harmonize and balance these factors.

The code started by importing the necessary libraries, importing: torch, torch.nn, torchvision, etc.
Next, the MLP design was initiated, starting by defining the number of layers, and neurons per layer from the input until the output layer, since the dataset used was MNIST, which deals with 28*28 pixels images, the number of input neurons was 784, the number kept decreasing by going to the next layers until the output layer was reached, which required 10 neurons demonstrating the numbers from 0 to 9, between each layer, the activation function was set, which in this case was ReLU, followed by setting the weight initialization to He initialization, both ReLU, and He results in decreasing the effect on overfitting, vanishing and exploding gradient to a certain extent, then the biases was set to 0, the next step was to prepare the raw data extracted from FashionMNIST dataset for usage and then uploading the dataset to the MLP, for optimization, Adam optimizer was used since it was most suitable for ReLU activation, next a CrossEntropyLoss was set for loss function and the code ends with a training loop, printing the loss for each epoch, and finally plotting the loss graph.


we mainly focused on finding the right harmony between different variables and experimenting with different paths to find the optimal solution, the results indicated were in the form of loss values for each epoch, and the execution time for the model, all of which indicated that the best path we reached was the first code displayed, with a final loss value of 0.1912, and runtime of 222.93 seconds.


> The research poster for this project can be found in the [BeyondAI Proceedings 2024](https://thinkingbeyond.education/beyondai_proceedings_2024/).
