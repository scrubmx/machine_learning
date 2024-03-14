defmodule ML.Classifiers.FizzBuzz do
  @moduledoc ~S"""
  Documentation for `ML.Classifiers.FizzBuzz`.
  """

  import ML.Activations, only: [relu: 1, softmax: 1]

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

  def gradient_descent(_vector, _weights, _label_vector, _current_loss) do
    # TODO: Implement the gradient descent algorithm
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
    [rem(x, 3), rem(x, 5)]
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

  defp one_hot_encode(number) do
    cond do
      rem(number, 3) == 0 -> [1, 0]
      rem(number, 5) == 0 -> [0, 1]
      true -> [0, 0]
    end
  end
end