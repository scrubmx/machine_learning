defmodule ML.Classifiers.FizzBuzz do
  @moduledoc ~S"""
  Documentation for `ML.Classifiers.FizzBuzz`.
  """

  import ML.Activations, only: [relu: 1, softmax: 1]

  @learning_rate 0.01

  @doc ~S"""
  Classify the given number into :fizz or :buzz.

  ### Error cases

      iex> ML.Classifiers.FizzBuzz.classify("hello")
      ** (ArgumentError) argument must be an integer, got: "hello"

  """
  def classify(number) when is_integer(number) do
    # TODO: The weights should be randomly generated
    weights = [
      [0.2, 0.3],
      [0.4, 0.1]
    ]

    vector = mods(number)

    label_vector = one_hot_encode(number)

    neural_network(vector, weights, label_vector)
  end

  def classify(number), do: raise(ArgumentError, message: "argument must be an integer, got: #{inspect(number)}")

  @doc ~S"""
  This function will take a vector and a matrix of weights and return
  the loss or cost function value. We'll keep iterating adjusting the
  weights until we find the weights that minimize the loss value.
  """
  def neural_network(vector, weights, label_vector, epsilon \\ 0.001) do
    predictions =
      vector_matrix_multiplication(vector, weights)
      |> softmax()
      |> Enum.map(&relu/1)

    current_loss = loss(predictions, label_vector)

    if current_loss <= epsilon do
      # IO.puts("Loss: #{current_loss}")
      # IO.inspect(predictions, label: "predictions", pretty: true)
      # IO.inspect(weights, label: "weights", pretty: true)
      decode_prediction(predictions)
    else
      # IO.puts("Loss: #{current_loss}")
      updated_weights = update_weights(vector, weights, label_vector, predictions)
      neural_network(vector, updated_weights, label_vector)
    end
  end

  def update_weights(vector, weights, label_vector, predictions) do
    # Calculate gradients matrix
    gradients =
      Enum.with_index(predictions)
      |> Enum.map(fn {pred, index} ->
        # Calculate gradient for each weight
        # Note: This is a simplified gradient calculation for demonstration.
        delta = 2 * (pred - Enum.at(label_vector, index))
        Enum.map(vector, &(&1 * delta))
      end)

    # Adjust weights by subtracting gradient scaled by learning rate
    new_weights =
      Enum.zip(weights, gradients)
      |> Enum.map(fn {weight_row, gradient_row} ->
        Enum.zip(weight_row, gradient_row)
        |> Enum.map(fn {w, g} -> w - @learning_rate * g end)
      end)

    new_weights
  end

  @doc ~S"""
  A gradient is a vector of partial derivatives.
  It points in the direction of the greatest rate of increase of the function.
  """
  def calculate_gradient(vector, weights, label_vector) do
  end

  @doc ~S"""
  This function will take a number and return a vector of
  the remainders when that number is divided by 3 and 5.

  ## Examples

      iex> ML.Classifiers.FizzBuzz.mods(3)
      [0, 3]

      iex> ML.Classifiers.FizzBuzz.mods(5)
      [2, 0]

      iex> ML.Classifiers.FizzBuzz.mods(15)
      [0, 0]
  """
  def mods(x) do
    if (rem(x, 15) == 0) do
      [0, 3] # Temporary hack to stop the neural network from breaking for "FizzBuzz"
    else
      [rem(x, 3), rem(x, 5)]
    end
  end

  def loss(predictions, label_vector) do
    Enum.zip(predictions, label_vector)
    |> Enum.map(fn {prediction, label_value} -> (prediction - label_value) ** 2 end)
    |> Enum.sum()
  end

  def loss_derivative(predictions, label_vector) do
    Enum.zip(predictions, label_vector)
    |> Enum.map(fn {prediction, label_value} -> 2 * (prediction - label_value) end)
  end

  @doc ~S"""
  Multiply a two matrices given using "weighed sum"
  which is also reffered to as the "dot product".
  """
  def vector_matrix_multiplication(vector, matrix) do
    Enum.map(matrix, fn row -> dot_product(vector, row) end)
  end

  @doc ~S"""
  In mathematics, the dot product is an algebraic operation that takes two equal-length
  sequences of numbers (usually coordinate vectors), and returns a single number.

  The dot product of two vectors:

    $${\displaystyle \mathbf {a} =[a_{1},a_{2},\cdots ,a_{n}]} \text{ and } {\displaystyle \mathbf {b} =[b_{1},b_{2},\cdots ,b_{n}]}$$

  Is specified with respect to an orthonormal basis, is defined as:

    $${\displaystyle \mathbf {a} \cdot \mathbf {b} =\sum _{i=1}^{n}a_{i}b_{i}=a_{1}b_{1}+a_{2}b_{2}+\cdots +a_{n}b_{n}}$$

  Where ${\displaystyle \Sigma}$ denotes summation and ${\displaystyle n}$ is the dimension of the vector space.

  ## References

    * [Dot Product on Wikipedia](https://en.wikipedia.org/wiki/Dot_product)

  ### Alternative Implementation

      vector
      |> Enum.with_index()
      |> Enum.reduce(0, fn {row, i}, acc -> acc + row * Enum.at(weights, i) end)

  """
  def dot_product(vector, weights) do
    Enum.zip(vector, weights)
    |> Enum.map(fn {value, weight} -> value * weight end)
    |> Enum.sum()
  end

  @doc ~S"""
  One hot encoding is a technique that we use to represent categorical variables as numerical values in a machine learning model.

  ## References

  [One-hot on Wikipedia](https://en.wikipedia.org/wiki/One-hot#Machine_learning_and_statistics)

  ## Example

  Consider the data where fruits, their corresponding categorical values, and prices are given:

    |  Fruit | Categorical value | Price |
    |:------:|:-----------------:|:-----:|
    |  Apple |         1         |   10  |
    |  Mango |         2         |   15  |
    | Orange |         3         |   20  |

  The output after applying one-hot encoding on the data is given as follows,

    | Apple | Mango | Orange | Price |
    |:-----:|:-----:|:------:|-------|
    |   1   |   0   |    0   |   10  |
    |   0   |   1   |    0   |   15  |
    |   0   |   0   |    1   |   20  |

  ### The advantages of using one hot encoding include:

  1. It allows the use of categorical variables in models that require numerical input.
  2. It can improve model performance by providing more information to the model about the categorical variable.
  3. It can help to avoid the problem of ordinality, which can occur when a categorical variable has a natural ordering (e.g. small, medium, large).

  ### The disadvantages of using one hot encoding include:

  1. It can lead to increased dimensionality, as a separate column is created for each category in the variable. This can make the model more complex and slow to train.
  2. It can lead to sparse data, as most observations will have a value of 0 in most of the one-hot encoded columns.
  3. It can lead to overfitting, especially if there are many categories in the variable and the sample size is relatively small.
  4. One-hot-encoding is a powerful technique to treat categorical data, but it can lead to increased dimensionality, sparsity, and overfitting. It is important to use it cautiously and consider other methods such as ordinal encoding or binary encoding.

  """
  def one_hot_encode(number) do
    cond do
      rem(number, 3) == 0 -> [1, 0]
      rem(number, 5) == 0 -> [0, 1]
      true -> [0, 0]
    end
  end

  def decode_prediction(predictions) do
    [fizz_prediction, buzz_prediction] = predictions # Example: [0.97234343, 0.024545]
    cond do
      fizz_prediction > buzz_prediction -> "Fizz"
      fizz_prediction < buzz_prediction -> "Buzz"
      true -> "Neither"
    end
  end
end
