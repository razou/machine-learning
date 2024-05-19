Logistic regression for image
The goal of this example is to build image classifier model using Logistic regression implemented from scratch.
- The implementation is mainly based on the power the vectorization of the `numpy` library
  - It avoids as much as possible the usage of loops (e.g., for loop, while loop, ...)

# Mathematical Formulation
- Let
  - $X$: training examples
  - $Y$: Labels for training examples in $X$
  - $m$: Number of training examples in $X$
  - $\alpha$:  learning rate
- For one training example
  - $z^{(i)}$
  - $\hat{y}^{(i)} = sigmoid(z^{(i)}) = \frac{1}{1 + \exp(-x^{(i)})}$
  - $Loss(y^{(i)}, \hat{y}^{(i)}) = y^{(i)} * \log(\hat{y}^{(i)} + (1 - y^{(i)}) * \log(1 - \hat{y}^{(i)}$
- The Cost function (for the whole training set): J = $\frac{1}{m}\sum Loss(y^{(i)}, hat{y}^{(i)})$
- Optimizer: gradient decent
  - $dW = \frac{\partial(J)}{\partial(W)}= \frac{1}{m} X(\hat{Y} - Y)^T$
  - $db = frac{\partial(J)}{\partial(b)} = \frac\sum(\hat{y}^{(i)}  - y^(i))$
  - The goal is to learn $W$ and $b$ by minimizing $J$:
    - After each iteration $W$ and $b$ are updated as following 
      - $W := W - \alpha * dW$
      - $b := b - \alpha * db$


# How to run/test it

- Install dependencies: `pip install -r requirements.txt`
- Then to `logistic_regression` directory 
- Run this command to get help: `python lr_model_train.py --help`
- Train logistic regression model: `lr_model_train.py`
  - You can specify parameters from command: `lr_model_train.py --learning_rate 0.01 --num_iterations 300`
- Optimization
  - Find the best value for the learning parameter: Run `python find_best_learning_rate.py`
- Make predictions
  - `python make_prediction.py`