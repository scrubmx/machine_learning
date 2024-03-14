defmodule ML.Classifiers.FizzBuzzTest do
  use ExUnit.Case

  alias ML.Classifiers.FizzBuzz

  doctest FizzBuzz

  test "mods/1 returns a list of the remainders of diving by 3 and 5" do
    assert FizzBuzz.mods(3) == [0, 3]
    assert FizzBuzz.mods(5) == [2, 0]
    assert FizzBuzz.mods(8) == [2, 3]
  end
end
