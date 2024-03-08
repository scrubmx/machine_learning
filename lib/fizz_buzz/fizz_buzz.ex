defmodule ML.FizzBuzz do
  @moduledoc false

  @learning_rate 0.01

  def classify(number) do
    # TODO: The weights should be randomly generated
    weights = [
      [0.2, 0.3],
      [0.4, 0.1]
    ]

    vector = mods(number)

    label_vector = one_hot_encode(number)

    neural_network(vector, weights, label_vector)
  end

  @doc ~S"""
  This function will take a vector and a matrix of weights and return
  the loss or cost function value. We'll keep iterating adjusting the
  weights until we find the weights that minimize the loss value.
  """
  def neural_network(vector, weights, label_vector, epsilon \\ 0.001) do
    probabilities =
      vector_matrix_multiplication(vector, weights)
      |> softmax()
      |> Enum.map(&relu/1)

    current_loss = loss(probabilities, label_vector)

    if current_loss <= epsilon do
      IO.puts("Loss: #{current_loss}")
      IO.inspect(probabilities, label: "Probabilities", pretty: true)
      current_loss
    else
      updated_weights = update_weights(vector, weights, label_vector, current_loss)
      neural_network(vector, updated_weights, label_vector)
    end
  end

  def update_weights(vector, weights, label_vector, current_loss) do
    gradient_descent(vector, weights, label_vector, current_loss)
  end

  def gradient_descent(vector, weights, label_vector, current_loss) do
    # TODO: Implement the gradient descent algorithm
  end

  @doc ~S"""
  This function will take a number and return a vector of
  the remainders when that number is divided by 3 and 5.

  ## Examples

    iex> ML.FizzBuzz.mods(3)
    [0, 3]
    iex> ML.FizzBuzz.mods(5)
    [2, 0]
    iex> ML.FizzBuzz.mods(15)
    [0, 0]
  """
  def mods(x) do
    [rem(x, 3), rem(x, 5)]
  end

  @doc ~S"""
  The **softmax function** takes as input a vector z of K real numbers, and normalizes it
  into a [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)
  consisting of K probabilities proportional to the exponentials of the input numbers.

  That is, prior to applying softmax, some vector components could be negative, or greater
  than one; and might not sum to 1; but after applying softmax, each component will be in the
  interval ${\displaystyle (0,1)$}, and the components will add up to 1, so that they can be
  interpreted as probabilities.

  For a vector ${\displaystyle z}$ of ${\displaystyle K}$ real numbers, the standard (unit) softmax function
  ${\displaystyle \sigma :\mathbb {R} ^{K}\mapsto (0,1)^{K}}$, where ${\displaystyle K\geq 1}$, is defined by the formula:

    $${\displaystyle \sigma (\mathbf {z} )_{i}={\frac {e^{z_{i}}}{\sum _{j=1}^{K}e^{z_{j}}}}\ \ {\text{ for }}i=1,\dotsc ,K{\text{ and }}\mathbf {z} =(z_{1},\dotsc ,z_{K})\in \mathbb {R} ^{K}.}$$

  ## Examples

    iex> probabilities = ML.FizzBuzz.softmax([0.75, 0.25, 0.25])
    [0.6, 0.2, 0.2]
    iex> Enum.sum(probabilities) == 1
    true
  """
  def softmax(vector) do
    sum = Enum.sum(vector)
    Enum.map(vector, fn x -> x / sum end)
  end

  @doc ~S"""
  Rectified Linear Unit (ReLU).
  Is an activation function defined as the positive part of its argument:

    $$f(x) = x^{+} = \max(0,x) = \frac{x + \left| x \right|}{2} = \left\{\begin{matrix} x & if x > 0, \\0 & otherwise. \end{matrix}\right.$$

  Where $x$ is the input to a neuron. This is also known as a ramp function.

  ## Examples

    iex> ML.FizzBuzz.relu(0.42)
    0.42

    iex> ML.FizzBuzz.relu(-0.25)
    0

  ## References

    * [Rectified Linear Unit on Wikipedia](https://en.wikipedia.org/wiki/Rectified_linear_unit)
    * [Activation Functions on Wikipedia](https://en.wikipedia.org/wiki/Activation_function)
  """
  def relu(x) do
    if x > 0, do: x, else: 0
  end

  def loss(predictions, label_vector) do
    Enum.zip(predictions, label_vector)
    |> Enum.map(fn {prediction, label_value} -> (prediction - label_value) ** 2 end)
    |> Enum.sum()
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

  ## Definition

    The dot product of two vectors
    $${\displaystyle \mathbf {a} =[a_{1},a_{2},\cdots ,a_{n}]} and {\displaystyle \mathbf {b} =[b_{1},b_{2},\cdots ,b_{n}]}$$,
    specified with respect to an orthonormal basis, is defined as:

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

  defp one_hot_encode(number) do
    cond do
      rem(number, 3) == 0 -> [1, 0]
      rem(number, 5) == 0 -> [0, 1]
      true -> [0, 0]
    end
  end
end
