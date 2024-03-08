defmodule ML do
  @moduledoc """
  Documentation for `ML`.
  """

  @doc """
  Fizz Buzz classification.
  The main entry point for the FizzBuzz classifier.
  Current  implementation only returns Fizz or Buzz.

  ## Examples

    iex> ML.fizz_buzz(3)
    :fizz
    iex> ML.fizz_buzz(5)
    :buzz
  """
  def fizz_buzz(number) do
    ML.FizzBuzz.classify(number)
  end
end
