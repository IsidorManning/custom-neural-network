import pandas as pd  # for reading the data
import numpy as np  # Linear Algebra

# Preparing the data

# Load the data into a pandas DataFrame
data_url = 'digit-recognizer-data/train.csv'
data_df = pd.read_csv(data_url)

# Load the data into a numpy array
data_np = np.array(data_df)
# Shuffle the data using numpy
np.random.shuffle(data_np)
# Get the shape of the dataset (number of rows & columns)
m, n = data_np.shape

# Split the data into x and y, features and labels. The first column contains all true
# values/labels for each instance.
x_data = data_np.T[:][1:n].T
y_data = data_np.T[:][0].T

# Split the data into train and test data for model evaluation
x_train = x_data[4200:m]
x_test = x_data[0:4200]
y_train = y_data[4200:m]
y_test = y_data[0:4200]

# Normalize all values in the x data to fall between 0 and 1 by dividing all
# values by 255.
x_train_norm = x_train / 255.
x_test_norm = x_test / 255.


def one_hot_encoder(inputs):
    """
    This function is a simple one-hot encoder used to encode certain features
    into binary by representing each of the categorical feature’s options as its
    own column with either a zero (for false) or a one (for true).

    param inputs: A numpy array
    return: The transformed, encoded array as a new numpy array
    """
    one_hot_encoded = np.zeros((y_train.size, np.max(inputs) + 1))
    for sample_id, y in enumerate(inputs):
        one_hot_encoded[sample_id][int(y)] = 1
    return one_hot_encoded


# One-hot encode all values in the y data.
y_train_one_hot = one_hot_encoder(y_train)
y_test_one_hot = one_hot_encoder(y_test)


# This class encapsulates one layer of neurons
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """
        In our constructor, we initialize our weights and biases;
        weights will be initialized randomly using the rand() function from NumPy.

        param n_inputs: the shape of the x-dimension (the number of rows)
        param n_neurons: the shape of the y-dimension (the number of columns)
        """
        # We subtract by 0.5 to make the distribution centered around zero
        self.weights = np.random.rand(n_inputs, n_neurons) - 0.5
        self.biases = np.random.rand(1, n_neurons) - 0.5
        self.a = None  # Activations
        self.z = None  # Weighted sum of inputs

    def forward(self, inputs):
        """
        Calculates one pass forward through a layer.

        param inputs: a matrix of inputs.
        return: None
        """
        self.z = np.dot(inputs, self.weights) + self.biases

    def relu(self, z):
        """
        Applies Rectified Linear Unit (ReLU) activation function element-wise.

        param z: Weighted sum of inputs
        return: None
        """
        self.a = np.maximum(0, z)

    @staticmethod
    def relu_deriv(z):
        """
        Computes the derivative of the ReLU activation function.

        param z: Weighted sum of inputs
        return: Derivative values
        """
        return np.where(z > 0, 1, 0)

    def softmax(self, z):
        """
        Applies the softmax activation function to the weighted sum of inputs.

        param z: Weighted sum of inputs
        return: None
        """
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.a = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class CategoricalCrossentropy:
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def calculate(self):
        """
        Calculates the categorical cross-entropy loss.

        return: Total loss value
        """
        sample_losses = self.forward()
        total_loss = np.mean(sample_losses)
        return total_loss

    def forward(self):
        """
        Computes the negative log-likelihoods for each sample.

        return: Array of negative log-likelihood values
        """
        y_pred_clipped = np.clip(self.y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(y_pred_clipped * self.y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


def feed_forward(data):
    """
    Forward pass through the entire defined neural network.

    param data: Input data
    return: Activations of the output layer
    """
    # Forward pass from the input data through the first hidden layer
    h1.forward(data)
    h1.relu(h1.z)

    # Forward pass from the second hidden layer through the output layer
    y_hat.forward(h1.a)
    y_hat.softmax(y_hat.z)

    return y_hat.a


def propagate_backward():
    """
    Backpropagation through the neural network.

    return: Gradients of weights and biases
    """
    batch_normalization_coefficient = 1 / y_train.size

    # Backpropagate the error through the output layer
    output_delta = y_hat.a - y_train_one_hot
    dWy = batch_normalization_coefficient * np.dot(h1.a.T, output_delta)
    dBy = batch_normalization_coefficient * np.sum(output_delta, axis=0, keepdims=True)

    # Backpropagate the error through the first hidden layer
    h1_delta = np.dot(output_delta, y_hat.weights.T) * h1.relu_deriv(h1.z)
    dW1 = batch_normalization_coefficient * np.dot(x_train_norm.T, h1_delta)
    dB1 = batch_normalization_coefficient * np.sum(h1_delta, axis=0, keepdims=True)

    return dW1, dB1, dWy, dBy


def gradient_descent(alpha):
    """
    Update weights and biases using gradient descent.

    param alpha: Learning rate
    return: None
    """
    # Back propagation
    dW1, dB1, dWy, dBy = propagate_backward()

    # Update the weights and biases using the gradients
    h1.weights -= alpha * dW1
    h1.biases -= alpha * dB1
    y_hat.weights -= alpha * dWy
    y_hat.biases -= alpha * dBy


def calculate_accuracy(y_pred, y_true):
    """
    Calculate the accuracy of the model predictions.

    param y_pred: Predicted labels
    param y_true: True labels
    return: Accuracy value
    """
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_true, axis=1)

    accuracy = np.mean(np.equal(y_pred_class, y_true_class))
    return accuracy


def serialize_epoch(i, loss, accuracy):
    """
    Serializes and prints one epoch neatly in a visually pleasing way.

    param i: The index of the current epoch
    param loss: The current loss of the model
    param accuracy: The current accuracy of the model
    return: None
    """
    epoch_repr = str(i + int(epochs * 0.05)).rjust(5)
    loss_repr = str(round(float(loss), 4)).ljust(8)
    accuracy_repr = str(round(float(accuracy), 4))
    print("epoch %s: Loss = %s Accuracy = %s" % (epoch_repr, loss_repr, accuracy_repr))


def train(n_iterations, alpha):
    """
    Train the neural network using gradient descent.

    param n_iterations: Number of training iterations
    param alpha: Learning rate
    return: None
    """
    for epoch in range(n_iterations):
        # Forward propagation
        predictions = feed_forward(x_train_norm)

        # Calculate the loss
        y_hat_loss = CategoricalCrossentropy(predictions, y_train_one_hot)
        loss = y_hat_loss.calculate()

        # Calculate the accuracy
        accuracy = calculate_accuracy(predictions, y_train_one_hot)

        # Perform gradient descent and update weights and biases
        gradient_descent(alpha)

        # Print the loss every 10 iterations
        if epoch % 10 == 0:            serialize_epoch(epoch, loss, accuracy)


# Defining the neural network's architecture
input_size = 784
hidden_size = 24
output_size = 10
h1 = DenseLayer(input_size, hidden_size)
y_hat = DenseLayer(hidden_size, output_size)

# Training
epochs = 2000
learning_rate = 0.10
train(epochs, learning_rate)
