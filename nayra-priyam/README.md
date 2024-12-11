![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2024.png)

# Limitations of Multilayer Perceptrons

The code goes through different versions of MLPs, testing one parameter at a time step by step. For each test, it measures loss values to see how each setup performs. This way, it also doubles as a way to catch mistakes during training.

The main goal was to compare loss values across epochs and see how they change with different settings. Testing activation functions, learning rates, and step sizes caused some changes in loss, but nothing too drastic because these are simple models. If we did the same tests on bigger, more complex MLPs, the losses would’ve probably gone up a lot more.

On the qualitative side, we looked at which activation functions, learning rates, and step sizes worked best. These choices affected how the models behaved, and the goal was to find a balance between them to get the best results.

The code starts by importing the usual libraries—torch, torch.nn, and torchvision. Then, it sets up the MLP structure. Since MNIST images are 28x28 pixels, the input layer has 784 neurons. The neuron count shrinks layer by layer until the output layer, which has 10 neurons (one for each digit). We tested activation functions like Sigmoid, ReLU, Tanh, and Leaky ReLU, and Sigmoid and ReLU performed the best. Weight initialization used He initialization to reduce overfitting and gradient issues, and biases were set to 0.

The FashionMNIST dataset was prepared and loaded into the model. We used the Adam optimizer because it works well with ReLU activation, and the loss function was CrossEntropyLoss. The training loop printed the loss after every epoch and plotted graphs to make comparisons easier.

Here’s how the testing went:

Activation Functions: We tried Sigmoid, ReLU, Tanh, and Leaky ReLU. Sigmoid and ReLU gave the lowest loss values.
Learning Rates: We tested 0.0001, 0.001, and 0.01, fixing the activation function to either Sigmoid or ReLU. The best learning rate for both was 0.001.
Step Sizes: We tested step sizes of 20, 30, and 40 while keeping the learning rate at 0.001 and trying both Sigmoid and ReLU.
After all the testing, the best combination we found was Sigmoid activation, a learning rate of 0.001, and a step size of 30. This gave a final loss value of 0.1734 and a runtime of 210.87 seconds. The testing process just showed how important it is to experiment and adjust things until you get the best setup.

> The research poster for this project can be found in the [BeyondAI Proceedings 2024](https://thinkingbeyond.education/beyondai_proceedings_2024/).
