defmodule ML do
  @moduledoc """
  Entry points for the machine learning examples.
  """

  alias ML.Classifiers.FizzBuzz

  @doc """
  Classifies an integer with the trained FizzBuzz network.

  Returns `"None"`, `"Fizz"`, `"Buzz"`, or `"FizzBuzz"`. See
  `ML.Classifiers.FizzBuzz` for the training method.

  ## Examples

      iex> ML.fizz_buzz(3)
      "Fizz"
      iex> ML.fizz_buzz(5)
      "Buzz"
      iex> ML.fizz_buzz(15)
      "FizzBuzz"
  """
  def fizz_buzz(number), do: FizzBuzz.classify(number)
end
