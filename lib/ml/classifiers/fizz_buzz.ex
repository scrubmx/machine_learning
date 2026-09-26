defmodule ML.Classifiers.FizzBuzz do
  @moduledoc """
  A small, deterministic neural network for the four FizzBuzz classes.

  The network has three inputs: a constant bias, whether the number is divisible
  by 3, and whether it is divisible by 5. A dense output layer has one neuron
  for each of `"None"`, `"Fizz"`, `"Buzz"`, and `"FizzBuzz"`. Softmax turns the
  four output scores into probabilities.

  `classify/1` starts with zero weights and trains on one example of each class
  (1, 3, 5, and 15). For each example, stochastic gradient descent subtracts
  `learning_rate * (predicted_probability - target) * input` from each weight.
  The target is a one-hot vector and this is the gradient of softmax cross
  entropy. After the fixed number of epochs, the network predicts the requested
  number. The requested number is never used as a training label.

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
    |> features()
    |> probabilities(weights)
    |> Enum.with_index()
    |> Enum.max_by(&elem(&1, 0))
    |> elem(1)
    |> then(&Enum.at(@labels, &1))
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
    initial_weights = for _ <- @labels, do: [0.0, 0.0, 0.0]

    Enum.reduce(1..@epochs, initial_weights, fn _, weights ->
      Enum.reduce(@training_examples, weights, fn {number, target}, current_weights ->
        input = features(number)
        predictions = probabilities(input, current_weights)

        current_weights
        |> Enum.with_index()
        |> Enum.map(fn {row, class} ->
          error = Enum.at(predictions, class) - if(class == target, do: 1.0, else: 0.0)

          Enum.zip_with(row, input, fn weight, feature ->
            weight - @learning_rate * error * feature
          end)
        end)
      end)
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

  defp probabilities(input, weights) do
    weights
    |> Enum.map(fn row -> Enum.zip_with(row, input, &*/2) |> Enum.sum() end)
    |> Activations.softmax()
  end
end
