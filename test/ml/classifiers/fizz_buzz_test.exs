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

    for {input, class} <- [{[1.0, 0.0, 0.0], 0}, {[1.0, 1.0, 0.0], 1}, {[1.0, 0.0, 1.0], 2}, {[1.0, 1.0, 1.0], 3}] do
      scores = Enum.map(weights, fn row -> Enum.zip_with(row, input, &*/2) |> Enum.sum() end)
      assert scores |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1) == class
      assert scores |> ML.Activations.softmax() |> Enum.at(class) > 0.9
    end
  end

  test "classify/1 rejects non-integers" do
    assert_raise ArgumentError, ~r/argument must be an integer/, fn ->
      FizzBuzz.classify(3.0)
    end
  end
end
