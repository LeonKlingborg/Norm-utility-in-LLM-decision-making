# Prompting norm utility reduces the volatility of LLM strategic decision-making

A repository containing code to document the methods in the paper *Prompting norm utility reduces the volatility of LLM strategic decision-making*. The code can be used to reproduce all prompts and the exact LLM configurations used to produce the results in the paper.

## System requirements

Running the models used in the paper ([Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B), [Gemma 4 31B it](https://huggingface.co/google/gemma-4-31B-it), and [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)) requires at least about 80 GB of video memory on modern graphics cards and about 100 GB of storage to keep model weights. We used two NVIDIA A100 GPUs when conducting our experiments. To run the models, we use [vLLM](https://docs.vllm.ai/en/v0.28.0/) through its offline Python API. Code dependencies are documented in [pyproject.toml](pyproject.toml).

## Quick-start example

To run a sample of example scenarios, first download the LLMs:

```bash
python download_models.py --model-root=/your/storage/directory
```

Second, to run the example sample:

```bash
python run_games.py --model-root=/your/storage/directory --output=/your/data/directory/results.csv --model=qwen3.8 --condition=with_belief_with_util
```

Running the code should generate a CSV containing the outcome of the scenario sample.

## Repository structure

The code provided here is split over the following files:

- [`download_models.py`](download_models.py) provides functionality for downloading the correct models and the exact snapshots used for experiments.

- [`vLLM_setup.py`](vLLM_setup.py) specifies model configurations and documents the configurations used for experiments in the paper.

- [`run_games.py`](run_games.py) provides code for running samples of examples of the scenarios explored in the paper.

## Data schema

The generated data file contains a table with the following columns:

- `condition`: label for the experimental condition.
- `benefit`: the benefit of at least one player cooperating in the game.
- `cost`: the total cost of cooperation (split between both players if both cooperate).
- `public_belief_proportion`: The proportion of the public that's said to believe cooperation is the right thing to do.
- `temperature`: The temperature parameter when sampling the next token from the log probs from the LLM.
- `prompt`: The prompt used.
- `raw_output`: The full output from the prompted model.
- `reasoning`: The reasoning part of the output.
- `answer`: The answer part of the output. The answer is intended to be formatted as a JSON string containing the actions recommended by the LLM. The intended keys of the JSON are `action` and `public_belief` with values in [`true`, `false`], where `true` indicates cooperation and belief in cooperation and `false` indicates defection and belief in defection.
