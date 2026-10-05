defmodule ML.Classifiers.FizzBuzz do
  @moduledoc """
  A small, deterministic neural network for the four FizzBuzz classes.

  The network has three inputs: a constant bias, whether the number is divisible
  by 3, and whether it is divisible by 5. A dense output layer has one neuron
  for each of `"None"`, `"Fizz"`, `"Buzz"`, and `"FizzBuzz"`. Softmax turns the
  four output scores into probabilities.

  `classify/1` starts with zero weights and trains on one example of each class
  (1, 3, 5, and 15). Each training step converts the number to features,
  predicts probabilities, calculates a gradient, and updates the weights. The
  loss for an example is the negative logarithm of its correct-class
  probability. `loss/2` makes this value available for inspection. The
  gradient for each weight is `(predicted_probability - target) * input`; this
  is the derivative of softmax cross entropy with a one-hot target. Stochastic
  gradient descent subtracts the gradient times the learning rate. After the
  fixed number of epochs, the network predicts the requested number. The
  requested number is never used as a training label.

  Training is intentionally repeated on every call so the function has no
  hidden state and gives the same result for the same input. This is a teaching
  example; the divisibility features already capture the FizzBuzz rule.
  """

  alias ML.Activations

  @labels ["None", "Fizz", "Buzz", "FizzBuzz"]
  @training_examples [{1, 0}, {3, 1}, {5, 2}, {15, 3}]
  @learning_rate 0.1
  @epochs 500

  @doc """
  Trains the network and returns its classification of an integer.

  The model is trained from scratch on four representative numbers for every
  call. It returns one of `"None"`, `"Fizz"`, `"Buzz"`, or `"FizzBuzz"`.

  ## Examples

      iex> ML.Classifiers.FizzBuzz.classify(1)
      "None"
      iex> ML.Classifiers.FizzBuzz.classify(9)
      "Fizz"
      iex> ML.Classifiers.FizzBuzz.classify(10)
      "Buzz"
      iex> ML.Classifiers.FizzBuzz.classify(30)
      "FizzBuzz"

  An integer is required:

      iex> ML.Classifiers.FizzBuzz.classify("hello")
      ** (ArgumentError) argument must be an integer, got: "hello"
  """
  def classify(number) when is_integer(number) do
    weights = train()

    number
    |> predict_probabilities(weights)
    |> decode_prediction()
  end

  def classify(number),
    do: raise(ArgumentError, "argument must be an integer, got: #{inspect(number)}")

  @doc """
  Trains and returns the four rows of output weights.

  Each row contains weights for `[bias, divisible_by_3, divisible_by_5]`.
  Rows correspond to `"None"`, `"Fizz"`, `"Buzz"`, and `"FizzBuzz"`, in that
  order. Training starts from zero weights and makes #{@epochs} passes over the
  four examples; it does not change global state.

  ## Example

      iex> weights = ML.Classifiers.FizzBuzz.train()
      iex> length(weights)
      4
      iex> Enum.all?(weights, &(length(&1) == 3))
      true
  """
  def train do
    Enum.reduce(1..@epochs, initial_weights(), fn _, weights -> train_epoch(weights) end)
  end

  @doc """
  Calculates cross-entropy loss for one prediction and its correct class.

  `predictions` is the four probabilities in the order `"None"`, `"Fizz"`,
  `"Buzz"`, `"FizzBuzz"`; `target` is the correct class index, from 0 to 3.
  A smaller loss means the network assigned more probability to the correct
  class. The training step uses the derivative of this loss directly, so it
  does not need to calculate the scalar loss before updating weights.

  ## Examples

      iex> ML.Classifiers.FizzBuzz.loss([0.25, 0.25, 0.25, 0.25], 1) |> Float.round(3)
      1.386

      iex> ML.Classifiers.FizzBuzz.loss([0.05, 0.9, 0.025, 0.025], 1) |> Float.round(3)
      0.105
  """
  def loss(predictions, target), do: -:math.log(Enum.at(predictions, target))

  @doc """
  Returns the four class probabilities for a number using supplied weights.

  The probabilities are ordered `"None"`, `"Fizz"`, `"Buzz"`, `"FizzBuzz"`.
  Use the weights from `train/0` to inspect what the network learned.

  ## Example

      iex> probabilities = ML.Classifiers.FizzBuzz.predict_probabilities(3, ML.Classifiers.FizzBuzz.train())
      iex> Enum.at(probabilities, 1) > 0.9
      true
  """
  def predict_probabilities(number, weights) when is_integer(number) do
    number |> features() |> forward_pass(weights)
  end

  defp initial_weights, do: for(_ <- @labels, do: [0.0, 0.0, 0.0])

  defp train_epoch(weights), do: Enum.reduce(@training_examples, weights, &train_example/2)

  defp train_example({number, target}, weights) do
    input = features(number)
    predictions = forward_pass(input, weights)
    gradients = calculate_gradients(input, predictions, target)
    update_weights(weights, gradients)
  end

  defp calculate_gradients(input, predictions, target) do
    predictions
    |> Enum.with_index()
    |> Enum.map(fn {prediction, class} ->
      gradient_for_class(input, prediction, class, target)
    end)
  end

  defp gradient_for_class(input, prediction, class, target) do
    error = prediction - one_hot_target(class, target)
    Enum.map(input, &(&1 * error))
  end

  defp one_hot_target(class, target), do: if(class == target, do: 1.0, else: 0.0)

  defp update_weights(weights, gradients) do
    Enum.zip_with(weights, gradients, &update_row/2)
  end

  defp update_row(row, gradient_row) do
    Enum.zip_with(row, gradient_row, fn weight, gradient ->
      weight - @learning_rate * gradient
    end)
  end

  @doc """
  Returns the remainders after division by 3 and 5.

  A zero in both positions identifies numbers divisible by 15, including zero.

  ## Examples

      iex> ML.Classifiers.FizzBuzz.mods(3)
      [0, 3]
      iex> ML.Classifiers.FizzBuzz.mods(5)
      [2, 0]
      iex> ML.Classifiers.FizzBuzz.mods(15)
      [0, 0]
  """
  def mods(number) when is_integer(number), do: [rem(number, 3), rem(number, 5)]

  defp features(number) do
    [mod_3, mod_5] = mods(number)
    [1.0, if(mod_3 == 0, do: 1.0, else: 0.0), if(mod_5 == 0, do: 1.0, else: 0.0)]
  end

  defp forward_pass(input, weights) do
    input |> output_scores(weights) |> Activations.softmax()
  end

  defp output_scores(input, weights), do: Enum.map(weights, &dot_product(input, &1))

  defp dot_product(left, right), do: Enum.zip_with(left, right, &*/2) |> Enum.sum()

  defp decode_prediction(predictions) do
    Enum.at(@labels, most_likely_class(predictions))
  end

  defp most_likely_class(predictions) do
    {_, class} = predictions |> Enum.with_index() |> Enum.max_by(&elem(&1, 0))
    class
  end
end
