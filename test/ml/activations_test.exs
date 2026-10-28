defmodule Ml.ActivationsTest do
  use ExUnit.Case, async: true

  doctest ML.Activations

  test "softmax produces normalized probabilities from logits" do
    [low, high] = ML.Activations.softmax([0, 1])
    assert low < high
    assert_in_delta low + high, 1.0, 1.0e-12
    assert ML.Activations.softmax([1000, 1000]) == [0.5, 0.5]
  end

  test "relu/1 returns the value if it's greatear than zero or zero" do
    assert ML.Activations.relu(1) == 1
    assert ML.Activations.relu(0) == 0
    assert ML.Activations.relu(-1) == 0
  end

  test "relu/1 works for a list of numbers" do
    assert ML.Activations.relu([1, 0, -1]) == [1, 0, 0]
  end
end
