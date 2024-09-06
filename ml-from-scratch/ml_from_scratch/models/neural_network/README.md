# Mathematical Formulation

Notations:

- $X$: Containing training examples, stacked by `column` (i.e. vertically)
  - $X \in \mathbb{R}^{n\times m}$ matrix (i.e, matrix of shape $(n, m)$ ), where $m$ corresponds to number of training examples and $n$ to number of features
  - Each column in $X$ has a shape of $(n, 1)$ and represents one training example.

$$
\mathbf{X} = \begin{bmatrix}
| & | & & | & & | \\
\mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(i)} & \cdots & \mathbf{x}^{(m)} \\
| & | & & | & & |
\end{bmatrix}
$$

- $L$: number of hidden layers
- For each hidden layer $l$ (from 1 to $L$), we note by:

  - $n^{[l]}$: Number of units in layer $l$
    - Where $n^{[0]}$: number of input features in $X$
  - $W^{[l]}:$ Weight matrix of shape $(n^{[l]}, n^{[l-1]})$
  - $b^{[l]}:$ Bias vector of shape $(n^{[l]}, 1)$
  - $Z^{[l]}:$ Linear combination (pre-activation) at layer $l$ of shape $(n^{[l]}, m)$
  - $A^{[l]}:$ Activation at layer $l$ of shape $(n^{[l]}, m)$
    - Should have the same dimension as the matrix $Z^{[l]}$
- $\mathcal{L}(A^{[L]}, Y)$ or $\mathcal{L}$: Loss function (evaluated on single training example)

  - Measures the error for a single training example
  - Used during the computation of gradients for a single training example in methods like stochastic gradient descent(SGD)

- $\mathcal{J}(W, b)$ or $\mathcal{J}$ : Cost function
  - Measures the average error over the entire training dataset.
  - Used to evaluate the overall performance of the model on the entire training set and is minimized during training.
  - $\mathcal{J}(W, b)=\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(A^{[L]\(i\)}, Y^{(i)})$
    - $A^{[L]}$: Predictions given by the output layer $L$

    
## Example

- Suppose that the size of the input matrix $X$ is $(12288, 209)$
  - $m=209$ training examples
  - $n_0=12288$

<table style="width:100%">
    <tr>
        <td>  </td>
        <td> <b>Shape of W</b> </td>
        <td> <b>Shape of b</b>  </td>
        <td> <b>Activation</b> </td>
        <td> <b>Shape of Activation</b> </td>
    <tr>
    <tr>
        <td> <b>Layer 1</b> </td>
        <td> $(n^{[1]},12288)$ </td>
        <td> $(n^{[1]},1)$ </td>
        <td> $Z^{[1]} = W^{[1]}  X + b^{[1]}$ </td>
        <td> $(n^{[1]},209)$ </td>
    <tr>
    <tr>
        <td> <b>Layer 2</b> </td>
        <td> $(n^{[2]}, n^{[1]})$  </td>
        <td> $(n^{[2]},1)$ </td>
        <td>$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$ </td>
        <td> $(n^{[2]}, 209)$ </td>
    <tr>
       <tr>
        <td> $\vdots$ </td>
        <td> $\vdots$  </td>
        <td> $\vdots$  </td>
        <td> $\vdots$</td>
        <td> $\vdots$  </td>
    <tr>  
   <tr>
       <td> <b>Layer L-1</b> </td>
        <td> $(n^{[L-1]}, n^{[L-2]})$ </td>
        <td> $(n^{[L-1]}, 1)$  </td>
        <td>$Z^{[L-1]} =  W^{[L-1]} A^{[L-2]} + b^{[L-1]}$ </td>
        <td> $(n^{[L-1]}, 209)$ </td>
   <tr>
   <tr>
       <td> <b>Layer L</b> </td>
        <td> $(n^{[L]}, n^{[L-1]})$ </td>
        <td> $(n^{[L]}, 1)$ </td>
        <td> $Z^{[L]} =  W^{[L]} A^{[L-1]} + b^{[L]}$</td>
        <td> $(n^{[L]}, 209)$  </td>
    <tr>
</table>

## Forward Propagation

For each layer $l$ from 1 to $L$:

- $Z^{[l]} = W^{[l]}A^{[l-1]} + b{[l]}$

  - $A^{[0]} = X$, the input data
- $A^{[l]} = \sigma^{[l]}(Z^{[l]})$

  - $\sigma^{[l]}$ is the activation function of layer $l$
    - Element-wise operation (i.e., apply $\sigma^{[l]}$ function to each element of $Z^{[l]}$ matrix )
- $Z^{[l]}$ and $A^{[l]}$ are two $\mathbb{R}^{n^{[l]}\times m}$ matrix (matrices of shape $(n^{[l]}, m)$ each)

  - $n^{[l]}$: number of units (or nodes) for the hidden `layer` $l$  (i.e., vertical indices correspond to hidden units or neurons)
  - $m$: number of training examples (i.e., horizontal indices correspond to training examples)

$$
\mathbf{Z^{[l]}} = \begin{bmatrix}
| & | & & | & & | \\
\mathbf{z}^{[l]\(1\)} & \mathbf{z}^{[l]\(2\)} & \cdots & \mathbf{z}^{[l]\(i\)} & \cdots & \mathbf{z}^{[l]\(m\)} \\
| & | & & | & & |
\end{bmatrix}
$$

$$
\mathbf{A^{[l]}} = \begin{bmatrix}
| & | & & | & & | \\
\mathbf{a}^{[l]\(1\)} & \mathbf{a}^{[l]\(2\)} & \cdots & \mathbf{a}^{[l]\(i\)} & \cdots & \mathbf{a}^{[l]\(m\)} \\
| & | & & | & & |
\end{bmatrix}
$$

where:

- Each column of $Z^{[l]}$ has a shape of $(n^{[l]}, 1)$ and represents the pre-activation values of all $n^{[l]}$ neurons in layer $l$ for a single training example. Equivalently each entry $Z^{[l]}[i, j]$ in the matrix $Z^{[l]}$  represents the pre-activation value for the $i^{th}$ neuron in layer $l$ for the $j^{th}$ training example.
  - Example: the column $z^{[l]\(2\)}$ corresponds the pre-activation values of all $n^{[l]}$ neurons in layer $l$ for the $2^{nd}$ training example.
    - $z^{[l]\(2\)} = (z^{[l]\(2\)}_{1}, z^{[l]\(2\)}_{2}, \ldots, z^{[l]\(2\)}_{n^{[l]}})^T$
- Each column of $A^{[l]}$ has a shape of $(n^{[l]}, 1)$ and represents the activations of layer $l$ for one training example. As for the matrix $Z^{[l]}$, each entry $A^{[l]}[i, j]$ in the matrix $A^{[l]}$ represents the activation value for the $i^{th}$ neuron in layer $l$ for the $j^{th}$ training example.

$$
z^{[l]\(i\)} =
    \begin{bmatrix}
        z^{[l]\(i\)}_{1} \\
        z^{[l]\(i\)}_{2} \\
        \vdots \\
        z^{[l]\(i\)}_{n^{[l]}}
    \end{bmatrix}
$$

$$
a^{[l]\(i\)} = \sigma^{[l]}(z^{[l]\(i\)}) =
   \begin{bmatrix}
        a^{[l]\(i\)}_{1} \\
        a^{[l]\(i\)}_{2} \\
        \vdots \\
        a^{[l]\(i\)}_{n^{[l]}}
    \end{bmatrix}
$$

## Backpropagation

- NB: As reminder
  - $A^{[0]} = X$, input data
  - $A^{[L]} = \hat{Y}$, predictions

1. For the output layer $L$

   - $dA^{[L]} = \frac{\partial \mathcal{L}(y, \hat{y})}{\partial A^{[L]}}$

   - $dZ^{[L]}= \frac{\partial \mathcal{L}(y, \hat{y})}{\partial Z^{[L]}}$

   - Example:
     - For cross-entropy loss function and sigmoid activation
       - $dA^{[L]} = [-\frac{y^{(1)}}{a^{(1)}} + \frac{1 - y^{(1)}}{1 -a^{(1)}}, -\frac{y^{(2)}}{a^{(2)}} + \frac{1 - y^{(2)}}{1 -a^{(2)}}, \ldots, -\frac{y^{(m)}}{a^{(m)}} + \frac{1 - y^{(m)}}{1 -a^{(m)}}]$, row vector of shape $(1, m)$
         - Numpy expression: $dA^{[L]} = - (np.divide(Y, A^{[L]}) - np.divide(1 - Y, 1 - A^{[L]}))$
       - $dZ^{[L]} = A^{[L]} - Y$ = $[a^{(1)} - y^{(1)}, a^{(2)} - y^{(2)}, \ldots, a^{(m)} - y^{(m)}]$, row vector of shape $(1, m)$
2. For each layer $l$ from $L-1$ to 1:

   1. Backpropagate the error and compute gradients

      - $dZ^{[l]} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} = \frac{\partial \mathcal{L}}{\partial A^{[l]}}.\frac{\partial A^{[l]}}{\partial Z^{[l]}} = dA^{[l]} \odot \sigma\prime^{[l]}(Z^{[l]})$
        - $dZ^{[l]}$ is a matrix of shape $(n^{[l]}, m)$, same dimension as $Z^{[l]}$
        - $\odot$ Denotes element-wise multiplication
        - $\frac{\partial \mathcal{L}}{\partial A^{[l]}} = dA^{[l]}$: Gradient of the loss with respect to the activations of layer $l$
        - $\frac{\partial A^{[l]}}{\partial Z^{[l]}} = \sigma\prime^{[l]}(Z^{[l]})$: Derivative of the activation function $\sigma^{[l]}$ with respect to the weighted input $z^{[l]}$ of layer $l$
      - $dW^{[l]} = \frac{\partial \mathcal{J}(W, b)}{\partial W^{[l]}} = \frac{1}{m}dZ^{[l]}A^{[l-1]^T}$
        - matrix of shape $(n^{[l]}, n^{[l-1]})$, same dimension as $W^{[l]}$
      - $db^{[l]} = \frac{\partial \mathcal{J}(W, b)}{\partial b^{[l]}} = \frac{1}{m}\sum_{i=1}^{m} dZ^{(i)[l]}$
        - $i$ denote the training sample example.
        - $\frac{\partial \mathcal{J}(W, b)}{\partial b^{[l]}}$ is a vector column of shape $(n^{[l]}, 1)$, same dimension as $b^{[l]}$.
        - Numpy expression for $db^{[l]}$ : $\frac{1}{m}np.sum(dZ^{[l]}, axis=1, keepdims=True)$
      - $dA^{[l-1]} = \frac{\partial \mathcal{J}(W, b)}{\partial A^{[l-1]}}= W^{[l]^T}dZ^{[l]}$
   2. Update parameters $W$ and $b$

      - $W^{[l]} = W^{[l]} - \eta\frac{\partial \mathcal{J}(W, b)}{\partial W^{[l]}} = W^{[l]} - \eta dW^{[l]}$
      - $b^{[l]} = b^{[l]} - \eta\frac{\partial \mathcal{J}(W, b)}{\partial b^{[l]}} = b^{[l]} - \eta db^{[l]}$

## Implementation

- Hyperparameters (they control the parameters $W$ and $b$)

  - Learning rate $\eta$ (some time noted by $\alpha$)
  - Number of iterations (of gradient decent)
  - Number of layers $L$ (or hidden layers $L-1$)
  - Size of hidden layers ($n^{[1]}, n^{[2]}, \ldots, n^{[L-1]}$)
  - Activation function (e.g., `sigmoid`,  `tanh`, `relu`, etc.,)
  - Etc.,
- For one iteration of the gradient decent

  1. For layer $l$
     1. Parameters

        1. $W^{[l]}$: Weights, matrix of shape $(n^{[l]}, n^{[l-1]})$
        2. $b^{[l]}$: Bias, vector of shape $(n^{[l]}, 1)$
     2. Forward propagation

        1. Input: $A^{[l-1]}$
        2. Compute $Z^{[l]} = W^{[l]}A^{[l-1]} + b{[l]}$
        3. Cache: $Z^{[l]}$, $W^{[l]}$, $b^{[l]}$
           - Used to store values computed during this step (i.e., forward propagation) to be used in the next one (i.e., backward propagation).
        4. Output: $A^{[l]} = \sigma^{[l]}(Z^{[l]})$
     3. Backpropagation

        - Input:
          - $dA^{[l]}$
          - $cache(Z^{[l]})$, $cache(W^{[l]})$, $cache(b^{[l]})$
            - e.g., $cache(Z^{[l]})$ computed and cached earlier from the forward propagation step
        - Compute:
          - $dZ^{[l]}$
        - Output:
          - $dA^{[l-1]}$
          - $dW^{[l]}$
          - $db^{[l]}$

## Misc


| **Activation Function** | **Forward Activation**                           | **Derivative of Activation**                | **Gradient for Output Layer $(\delta^{[L]})$** | **Gradient for Hidden Layers $(\delta^{[l]})$**                                |
| ----------------------- |--------------------------------------------------|---------------------------------------------|------------------------------------------------|--------------------------------------------------------------------------------|
| **Sigmoid**             | $$\sigma(z) = \frac{1}{1 + e^{-z}}$$             | $$\sigma(z)(1 - \sigma(z))$$                | $$\delta^{[L]} = A^{[L]} - Y$$                 | $$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \cdot A^{[l]}(1 - A^{[l]})$$     |
| **ReLU**                | $$\text{ReLU}(z) = \max(0, z)$$                  | $$\begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$ | $$\delta^{[L]} = A^{[L]} - Y$$                 | $$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \cdot \mathbf{1}_{Z^{[l]} > 0}$$ |
| **Tanh**                | $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$ | $$1 - \tanh^2(z)$$                          | $$\delta^{[L]} = A^{[L]} - Y$$                 | $$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \cdot (1 - (A^{[l]})^2)$$        |


## Example of use case


1. Install `poetry`
   - `https://python-poetry.org/docs/#installation`
2. Install dependencies 
    - Run `poetry install` command from `ml-from-scratch` directory (which contains `pyproject.toml` file.) 
3. Train Neural Network model
   - Run `python train.py --help` from `machine-learning/ml-from-scratch/ml_from_scratch/models/neural_network` directory, to get help on details on parameters
   - Example: Train model en verbose mode (i.e. print logs, and some metadatas), with 1000 epochs, evaluate it and save the model artefact at the end of training process. 
     - `python train.py --verbose True --evaluate_model True --num_iterations 1000 --save_model True`
