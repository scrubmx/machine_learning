defmodule ML.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :m_l,
      name: "Machine Learning",
      version: @version,
      elixir: "~> 1.16",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:ex_doc, "~> 0.31", only: :dev, runtime: false}
    ]
  end

  # Run "mix help docs" to learn about documentation.
  defp docs do
    [
      main: "README",
      source_ref: "v#{@version}",
      logo: nil,
      source_url: "https://github.com/scrubmx/machine_learning",
      extras: [
        "README.md",
        "notebooks/livebooks.md",
        "notebooks/classifiers/fizz_buzz.livemd"
      ],
      groups_for_extras: [
        Classification: Path.wildcard("/classification/*.livemd")
      ],
      groups_for_modules: [
        Main: [
          ML
        ],
        Activations: [
          ML.Activations
        ],
        Classifiers: [
          ML.Classifiers.FizzBuzz
        ]
      ],
      before_closing_head_tag: {Docs, :before_closing_head_tag, []},
      before_closing_body_tag: {Docs, :before_closing_body_tag, []}
    ]
  end
end
