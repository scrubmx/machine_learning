defmodule ML.Classifiers.FizzBuzzTest do
  use ExUnit.Case, async: true

  alias ML.Classifiers.FizzBuzz

  doctest FizzBuzz

  test "mods/1 returns the unmodified remainders, including FizzBuzz" do
    assert FizzBuzz.mods(3) == [0, 3]
    assert FizzBuzz.mods(5) == [2, 0]
    assert FizzBuzz.mods(8) == [2, 3]
    assert FizzBuzz.mods(15) == [0, 0]
  end

  test "classify/1 predicts every divisibility combination in a complete cycle" do
    for number <- 0..14 do
      expected =
        cond do
          rem(number, 15) == 0 -> "FizzBuzz"
          rem(number, 3) == 0 -> "Fizz"
          rem(number, 5) == 0 -> "Buzz"
          true -> "None"
        end

      assert FizzBuzz.classify(number) == expected
    end
  end

  test "classify/1 works beyond the training numbers, including negatives" do
    assert FizzBuzz.classify(1_000_002) == "Fizz"
    assert FizzBuzz.classify(-50) == "Buzz"
    assert FizzBuzz.classify(-45) == "FizzBuzz"
    assert FizzBuzz.classify(-7) == "None"
  end

  test "train/0 performs deterministic gradient updates" do
    weights = FizzBuzz.train()

    assert length(weights) == 4
    assert Enum.all?(weights, &(length(&1) == 3))
    assert Enum.any?(List.flatten(weights), &(&1 != 0.0))
    assert FizzBuzz.train() == weights

    for {number, class} <- [{1, 0}, {3, 1}, {5, 2}, {15, 3}] do
      predictions = FizzBuzz.predict_probabilities(number, weights)

      assert predictions |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1) == class
      assert Enum.at(predictions, class) > 0.9
      assert FizzBuzz.loss(predictions, class) < FizzBuzz.loss([0.25, 0.25, 0.25, 0.25], class)
    end
  end

  test "loss/2 falls as the correct class gets more probability" do
    assert FizzBuzz.loss([0.05, 0.9, 0.025, 0.025], 1) <
             FizzBuzz.loss([0.25, 0.25, 0.25, 0.25], 1)
  end

  test "classify/1 rejects non-integers" do
    assert_raise ArgumentError, ~r/argument must be an integer/, fn ->
      FizzBuzz.classify(3.0)
    end
  end
end
