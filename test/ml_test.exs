defmodule MLTest do
  use ExUnit.Case, async: true

  # doctest ML

  describe "fizz_buzz" do
    @tag :skip
    test "predicts :fizz for numbers that are divisible by 3" do
      assert ML.fizz_buzz(3) == :fizz
    end

    @tag :skip
    test "predicts :buzz for numbers that are divisible by 5" do
      assert ML.fizz_buzz(5) == :buzz
    end
  end
end
