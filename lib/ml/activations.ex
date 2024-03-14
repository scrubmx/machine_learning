defmodule ML.Activations do
  @moduledoc """
  Activation functions are element-wise, (typically) non-linear functions
  called on the output of another layer, such as a dense layer:

      number
      |> softmax(vector)
      |> &relu/1

  Activation functions output the "activation" or how active a given layer's
  neurons are in learning a representation of the data-generating distribution.

  Some activations are commonly used as output activations. For example 
  `softmax` is often used as the output in multiclass classification problems
  because it returns a categorical probability distribution:

      iex> ML.Activations.softmax([1, 2, 3])
      [0.16666666666666666, 0.3333333333333333, 0.5]

  Other activations such as `sigmoid` are used because they have desirable
  properties, such as keeping the output tensor constrained within a range.

  Generally, the choice of activation function is arbitrary; although some
  activations work better than others in certain problem domains. For example
  ReLU (rectified linear unit) activation is a widely-accepted default. You can
  see a list of activation functions and implementations [here](https://paperswithcode.com/methods/category/activation-functions).
  """

  @doc ~S"""
  The **softmax function** takes as input a vector z of K real numbers, and normalizes it
  into a [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)
  consisting of K probabilities proportional to the exponentials of the input numbers.

  That is, prior to applying softmax, some vector components could be negative, or greater
  than one; and might not sum to 1; but after applying softmax, each component will be in the
  interval ${\displaystyle (0,1)$}, and the components will add up to 1, so that they can be
  interpreted as probabilities.

  For a vector ${\displaystyle z}$ of ${\displaystyle K}$ real numbers, the standard (unit) softmax function
  ${\displaystyle \sigma :\mathbb {R} ^{K}\mapsto (0,1)^{K}}$, where ${\displaystyle K\geq 1}$, is defined by the formula:

    $${\displaystyle \sigma (\mathbf {z} )_{i}={\frac {e^{z_{i}}}{\sum _{j=1}^{K}e^{z_{j}}}}\ \ {\text{ for }}i=1,\dotsc ,K{\text{ and }}\mathbf {z} =(z_{1},\dotsc ,z_{K})\in \mathbb {R} ^{K}.}$$

  ## Examples

      iex> probabilities = ML.Activations.softmax([0.75, 0.25, 0.25])
      [0.6, 0.2, 0.2]
      iex> Enum.sum(probabilities) == 1
      true
  """
  def softmax(vector) do
    sum = Enum.sum(vector)
    Enum.map(vector, fn x -> x / sum end)
  end

  @doc ~S"""
  Rectified Linear Unit (ReLU).
  Is an activation function defined as the positive part of its argument:

    $$f(x) = x^{+} = \max(0,x) = \frac{x + \left| x \right|}{2} = \left\{\begin{matrix} x & if x > 0, \\0 & otherwise. \end{matrix}\right.$$

  Where $x$ is the input to a neuron. This is also known as a ramp function.

  ## Examples

      iex> ML.Activations.relu(0.42)
      0.42

      iex> ML.Activations.relu(-0.25)
      0

      iex> ML.Activations.relu([1, -2, 3])
      [1, 0, 3]

  ## References

    * [Rectified Linear Unit on Wikipedia](https://en.wikipedia.org/wiki/Rectified_linear_unit)
    * [Activation Functions on Wikipedia](https://en.wikipedia.org/wiki/Activation_function)
  """
  def relu(x) when is_number(x), do: max(0, x)
  def relu(x) when is_list(x), do: x |> Enum.map(&relu/1)
  def relu(x), do: x
end
