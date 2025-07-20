# Notes

## How AI Models Learn: Training, Neural Networks, and Generalization

### Training in AI

Training an AI (or machine learning) model means exposing it to large amounts of data so it can learn to perform a task. In practice, this involves feeding the model labeled examples, measuring its errors, and adjusting its parameters (weights and biases) to minimize those errors. AI model training refers to this process: feeding the algorithm data, examining the results, and tweaking the model output to increase accuracy and efficacy. In other words, training is an iterative process of "digesting" datasets to fit model coefficients and improve predictions. For example, training a cat-vs-dog image classifier means showing it thousands of labeled cat/dog pictures and using optimization (like gradient descent) to adjust weights so its predictions on new images are as accurate as possible.

### Overview of Neural Networks

An artificial neural network (ANN) is a computational model loosely inspired by the brain's network of neurons.

This shows a basic "neuron" or node: it computes a weighted sum of inputs and passes it through an activation function. A full network consists of layers of such neurons. Inputs (features) enter the network, propagate forward through one or more hidden layers, and produce outputs (predictions) at the final layer. Each connection has a weight (analogous to a synapse strength) that multiplies the input signal. The activation function (e.g. sigmoid, ReLU, softmax) adds nonlinearity so the network can model complex patterns. According to Wikipedia, an ANN is "inspired by the structure and functions of biological neural networks". In training, all these weights are adjusted to minimize the difference between the network's outputs and the true targets. In summary, a neural network is a parameterized function with a layered structure of weighted sum → activation operations, trained to fit input-output data.

### How Training Resembles the Human Brain

AI learning is often likened to how humans learn. In both cases, repeated exposure to examples causes connection strengths to change. In backpropagation (the main training algorithm), errors flow backward through the network to adjust weights – loosely analogous to how a brain might strengthen or weaken synaptic connections when learning from mistakes. However, there are important differences: humans can often learn a new concept from a single example and retain previous knowledge, whereas artificial networks usually require many examples and can suffer from catastrophic forgetting of old information when trained on new tasks. For instance, one study notes: "we can learn new information by just seeing it once, while artificial systems need to be trained hundreds of times… [and] learning new information in artificial neural networks often interferes with existing knowledge". In summary, ANNs borrow the idea of neuron-like units and synaptic weights from biology, but current training methods (e.g. gradient-based optimization) are still quite different from the brain's very efficient, one-shot learning processes.

## Deep Learning Basics: Layers, Activation, Forward/Backward Propagation

**Reference:** `https://playground.tensorflow.org`

Deep learning creates bigger neural networks with multiple layers stacked on top of each other. Think of it like a multi-story building where each floor processes information before passing it to the next floor.

### Network Structure

A typical network has three main parts:

- **Input layer:** Where data enters the network
- **Hidden layers:** The middle floors that do the processing work
- **Output layer:** Where the final answer comes out

### Activation Functions

Each layer uses special functions called activation functions. These are like switches that decide whether information should be passed along or not. Here are the common ones:

- **ReLU:** Acts like a one-way valve - it only lets positive values through and blocks negative ones. This helps deep networks learn better.
- **Sigmoid:** Creates a smooth S-shaped curve that squashes any input into a value between 0 and 1. Think of it like a dimmer switch.
- **Tanh:** Similar to sigmoid but squashes values between -1 and 1 instead.
- **Softmax:** Used at the end to convert the network's raw scores into probabilities. If you're classifying cats vs dogs vs birds, softmax makes sure the three probabilities add up to 100%.

### Why Activation Functions Matter

Without these activation functions, even a deep network would act like a simple straight line. The activation functions let the network learn curved and complex patterns in data - like recognizing that a cat's face is round while a dog's might be more elongated.

### How the Network Works

**Forward Propagation** is how information flows through the network:

1. Data enters at the input layer
2. Each layer takes the information from the previous layer
3. It combines this information using weights (like importance scores)
4. Adds a bias (like a baseline adjustment)
5. Applies the activation function
6. Passes the result to the next layer
7. This continues until reaching the output

### Measuring Success

After the network makes a prediction, we need to check how good it was. This is done with a **Loss Function** - think of it as a report card that gives the network a single score for how wrong it was.

- **For predicting numbers (like house prices):** We use Mean Squared Error, which basically looks at how far off each prediction was, squares those distances, and averages them.
- **For classification (like identifying cats vs dogs):** We use cross-entropy loss, which heavily penalizes confident wrong answers more than uncertain wrong answers.

The loss function gives us one number that tells us "how bad" the network's guesses were. The goal is to make this number as small as possible through training.

### Training Algorithm (Pseudocode)

A typical training loop looks like:

for each epoch:
for each batch of training data:
# Forward pass: compute model outputs
outputs = model.forward(batch_inputs)
loss = Loss(outputs, batch_labels)
# Backward pass: compute gradients
gradients = model.backward(loss)
# Update weights (gradient descent step)
for each weight w:
w = w - learning_rate * gradients[w]


This iteratively reduces the loss by adjusting weights. As noted in literature, "gradient-based methods such as backpropagation are usually used to estimate the parameters of the network". In effect, training deep networks is just iterated forward evaluations and gradient steps to minimize the chosen loss.

## Data, Loss Functions, and Optimization Methods

### Preparing Your Data

When working with data, you need to split it into three groups:

- **Training set:** This is like the textbook your network studies from. It learns patterns from this data.
- **Validation set:** This is like practice tests. You use this to fine-tune your network's settings without cheating.
- **Test set:** This is the final exam. You only use this once at the very end to see how well your network really performs on completely new data.

The whole point is to create a network that works well on data it has never seen before. If your network only works on the training data but fails on new data, it's like a student who memorized answers but doesn't understand the concepts.

### Choosing the Right Loss Function

Different problems need different ways to measure mistakes:

**For predicting numbers (regression):**
- Mean Squared Error heavily punishes big mistakes
- Mean Absolute Error treats all mistakes more equally

**For classification (picking categories):**
- Cross-entropy loss works well for choosing between different classes
- It's especially good because it punishes confident wrong answers more than uncertain ones

The loss function doesn't just measure how wrong the network is - it also guides the learning process by showing which direction to improve.

### Training the Network (Optimization)

Training a network is like teaching someone to ride a bike by making small adjustments each time they wobble. This process is called gradient descent.

- **Basic Gradient Descent:** Look at the mistake, figure out which direction would reduce it, then make a small adjustment in that direction. Repeat this thousands of times.
- **Stochastic Gradient Descent (SGD):** Instead of looking at all the data at once, look at just one example (or a small group) at a time. This is faster and often works better.

**Advanced Methods:**
- **Momentum:** Like a ball rolling downhill, it builds up speed in the right direction and doesn't get stuck easily
- **Adam:** Automatically adjusts how big steps to take for each part of the network individually

### Finding the Right Learning Speed

The learning rate is like how big steps the network takes when improving:

- **Too big steps:** The network might overshoot the best answer and never settle down
- **Too small steps:** Learning takes forever
- **Just right:** The network learns efficiently and finds good solutions

Most modern training uses smart methods that automatically adjust the step size or add momentum to make learning faster and more reliable.

## Overfitting, Underfitting, and Generalization

Balancing model complexity is crucial. A too-simple model may underfit: it fails to capture the underlying pattern even on the training data (high bias). In contrast, a too-complex model may overfit: it fits the training data (including noise) extremely well but performs poorly on new data (high variance). The diagram below illustrates these regimes:

According to sources, "underfitting occurs when the model fails to capture the underlying trend in the data", while "overfitting occurs when the model learns the training data too well…performing well on training data but poorly on unseen data". The ultimate goal is good generalization: strong performance on new inputs. As one reference notes, the main objective in supervised learning is "to build a model that will be able to make accurate predictions on new, unseen data".

Preventing overfitting often involves techniques like regularization (L1/L2 weight penalties), dropout (randomly deactivating neurons during training), or using more data. Early stopping (halting training when validation error stops improving) is also common. If a model is underfitting, one can increase its capacity (more layers or neurons) or input more features. Monitoring training vs. validation error curves is key: underfitting shows high error on both, whereas overfitting shows low training error but rising validation error. In practice, one strives for a model just complex enough to fit the true pattern but no more – the "sweet spot" that yields the lowest validation/test error and hence the best generalization.

In summary: AI model training is the iterative process of adjusting network weights on large datasets to minimize a defined loss. Neural networks, inspired by the brain's neurons, use many layers of weighted units and nonlinear activations to model complex functions. During training, inputs are propagated forward, errors are propagated backward to compute gradients, and optimization algorithms update the weights. Proper training requires suitable data splits, effective loss functions, and careful optimization to avoid overfitting and achieve good generalization.
