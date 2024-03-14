defmodule Ml.ActivationsTest do
  use ExUnit.Case, async: true

  doctest ML.Activations

  test "relu/1 returns the value if it's greatear than zero or zero" do
    assert ML.Activations.relu(1) == 1
    assert ML.Activations.relu(0) == 0
    assert ML.Activations.relu(-1) == 0
  end

  test "relu/1 works for a list of numbers" do
    assert ML.Activations.relu([1, 0, -1]) == [1, 0, 0]
  end
end
