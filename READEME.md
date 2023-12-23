# Simulação de Redes Neurais em Python

Este é um projeto simples para simular redes neurais em Python usando NumPy. A implementação inclui uma classe `NeuralNetwork` que permite a criação, treinamento e teste de uma rede neural com uma camada oculta. Além disso, foram incorporadas algumas melhorias, como a função de ativação ReLU, termo de momento (momentum) e regularização L2.

## Funcionalidades

- Implementação de uma rede neural com uma camada oculta.
- Escolha entre funções de ativação ReLU ou Sigmoid.
- Incorporação de termo de momento (momentum) para acelerar o treinamento.
- Adição de regularização L2 para evitar overfitting.

## Como Usar

1. **Instalação:**
   Certifique-se de ter o NumPy instalado. Caso não tenha, você pode instalá-lo usando:
   ```bash
   pip install numpy

### Uso Básico:

1. Crie uma instância da classe NeuralNetwork com os parâmetros desejados, como número de entradas, neurônios na camada oculta e número de saídas.
2. Escolha a função de ativação desejada ('relu' ou 'sigmoid').
3. Treine a rede neural usando o método train com dados de entrada e rótulos desejados.

```python
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation_function='relu')
nn.train(inputs, targets, epochs=10000, learning_rate=0.1)
```
### Teste da Rede Neural Treinada:

Use o método forward para obter as saídas previstas para novos dados.

```python
test_input = np.array([[0, 0]])
predicted_output = nn.forward(test_input)
print(f'Input: {test_input}, Predicted Output: {predicted_output}')
```
## Parâmetros Personalizáveis

`input_size`: Número de entradas da rede neural.

`hidden_size`: Número de neurônios na camada oculta.

`output_size`: Número de saídas da rede neural.

`activation_function`: Função de ativação para a camada oculta ('relu' ou 'sigmoid').

`regularization_strength`: Força da regularização L2 para evitar overfitting.

`momentum`: Termo de momento para acelerar a convergência durante o treinamento.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir problemas, propor melhorias ou enviar solicitações de pull.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para mais detalhes.