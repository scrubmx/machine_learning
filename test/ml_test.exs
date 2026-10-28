defmodule MLTest do
  use ExUnit.Case, async: true

  doctest ML

  describe "fizz_buzz" do
    test "predicts Fizz for numbers divisible by 3" do
      assert ML.fizz_buzz(3) == "Fizz"
    end

    test "predicts Buzz for numbers divisible by 5" do
      assert ML.fizz_buzz(5) == "Buzz"
    end

    test "predicts FizzBuzz and None" do
      assert ML.fizz_buzz(15) == "FizzBuzz"
      assert ML.fizz_buzz(7) == "None"
    end
  end
end
