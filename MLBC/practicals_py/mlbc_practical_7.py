import numpy as np

# 1. Define the activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Prepare XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# 3. Initialize network parameters
np.random.seed(42)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))

learning_rate = 0.1
epochs = 10000

# 4. Train the neural network
print("Initiating neural network training...")

for epoch in range(epochs):

    # Forward propagation
    hidden_layer_input = np.dot(X, hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, output_weights)
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    error = y - predicted_output

    output_gradient = error * sigmoid_derivative(predicted_output)

    hidden_error = output_gradient.dot(output_weights.T)
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Update weights
    output_weights += hidden_layer_output.T.dot(output_gradient) * learning_rate
    hidden_weights += X.T.dot(hidden_gradient) * learning_rate

print("Training complete.")

# 5. Display final prediction
print("\nFinal predicted output:")

print(predicted_output.round(2))

# 6. Convert probabilities into binary predictions
binary_output = (predicted_output >= 0.5).astype(int)

print("\nBinary predicted output:")
print(binary_output)

print("\nActual output:")
print(y)