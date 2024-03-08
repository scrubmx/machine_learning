defmodule ML.FizzBuzzTest do
  use ExUnit.Case

  doctest ML.FizzBuzz

  test "mods/1 returns a list of the remainders of diving by 3 and 5" do
    assert ML.FizzBuzz.mods(3) == [0, 3]
    assert ML.FizzBuzz.mods(5) == [2, 0]
    assert ML.FizzBuzz.mods(8) == [2, 3]
  end

  test "relu/1 returns the value if it's greatear than zero or zero" do
    assert ML.FizzBuzz.relu(1) == 1
    assert ML.FizzBuzz.relu(0) == 0
    assert ML.FizzBuzz.relu(-1) == 0
  end
end
