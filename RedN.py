import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_function='relu', regularization_strength=0.01, momentum=0.9):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        self.activation_function = activation_function
        self.regularization_strength = regularization_strength
        self.momentum = momentum
        self.prev_update_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.prev_update_weights_hidden_output = np.zeros_like(self.weights_hidden_output)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        if self.activation_function == 'relu':
            self.hidden_output = self.relu(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        elif self.activation_function == 'sigmoid':
            self.hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        else:
            raise ValueError("Activation function not supported")

        self.final_output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        return self.final_output

    def backward(self, inputs, targets, learning_rate):
        error = targets - self.final_output

        if self.activation_function == 'relu':
            output_delta = error * self.relu_derivative(self.final_output)
            hidden_delta = output_delta.dot(self.weights_hidden_output.T) * self.relu_derivative(self.hidden_output)
        elif self.activation_function == 'sigmoid':
            output_delta = error * self.sigmoid_derivative(self.final_output)
            hidden_delta = output_delta.dot(self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_output)
        else:
            raise ValueError("Activation function not supported")

        # Update weights and biases with momentum and regularization
        self.prev_update_weights_hidden_output = (
            self.momentum * self.prev_update_weights_hidden_output +
            self.hidden_output.T.dot(output_delta) * learning_rate -
            self.regularization_strength * self.weights_hidden_output
        )
        self.weights_hidden_output += self.prev_update_weights_hidden_output

        self.prev_update_weights_input_hidden = (
            self.momentum * self.prev_update_weights_input_hidden +
            inputs.T.dot(hidden_delta) * learning_rate -
            self.regularization_strength * self.weights_input_hidden
        )
        self.weights_input_hidden += self.prev_update_weights_input_hidden

        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(inputs)
            self.backward(inputs, targets, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(targets - output))
                print(f'Epoch {epoch}, Loss: {loss}')

# Exemplo de uso
if __name__ == "__main__":
    # Dados de entrada
    inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

    # Rótulos desejados
    targets = np.array([[0], [1], [1], [0]])

    # Criação da rede neural com ReLU como função de ativação
    nn_relu = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation_function='relu')

    # Treinamento da rede neural com ReLU
    print("Training with ReLU:")
    nn_relu.train(inputs, targets, epochs=10000, learning_rate=0.1)

    # Teste da rede neural treinada com ReLU
    test_input = np.array([[0, 0]])
    predicted_output = nn_relu.forward(test_input)
    print(f'Input: {test_input}, Predicted Output: {predicted_output}')

    # Criação da rede neural com Sigmoid como função de ativação
    nn_sigmoid = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation_function='sigmoid')

    # Treinamento da rede neural com Sigmoid
    print("\nTraining with Sigmoid:")
    nn_sigmoid.train(inputs, targets, epochs=10000, learning_rate=0.1)

    # Teste da rede neural treinada com Sigmoid
    test_input = np.array([[0, 0]])
    predicted_output = nn_sigmoid.forward(test_input)
    print(f'Input: {test_input}, Predicted Output: {predicted_output}')
