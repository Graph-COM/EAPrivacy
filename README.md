# [ICLR 2026] EAPrivacy - Measuring Physical-World Privacy Awareness of Large Language Models

[![arXiv](https://img.shields.io/badge/📄arXiv-2510.02356-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e)](https://arxiv.org/abs/2510.02356)
![Theme Image](./assets/theme_pic.png)

This repository contains the code for generating scenarios and evaluating Large Language Models (LLMs) for the EAPrivacy benchmark.

## Quick Start

EAPrivacy uses *seeds* to generate scenarios, each reflecting key design principles and characteristics of its tier. Scenario seeds are organized by tier (tier1 to tier4) in the [eai_bench](./eai_bench/) folder.

## Tier 1
- [Tier 1 Seeds](./eai_bench/tier_1.json)

Tier 1 focuses on identifying inherently sensitive objects in physical environments, testing foundational privacy knowledge. The original information for Tier 1 seeds is sourced from [ConfAIde](https://arxiv.org/abs/2310.17884). Common collocations of sensitive information attached to objects are listed under the `physical objects` field, and combinations for scenario generation are provided in the `combinations` field, as described in the paper.

## Tier 2
- [Tier 2 Seeds](./eai_bench/tier_2.json)

Tier 2 evaluates actions under changing environmental conditions, testing adaptability to dynamic privacy requirements. Each scenario (`scenario_name`) includes multiple environmental states (`environment_states`) and multimodal cues (`perception_cues`). A shared candidate action pool is provided, with ground truth and human-rated labels in `candidate_action_ratings`.

Example of Tier 2 ground truth and human labels:
```json
{
    "action": "knock_door('door_a')",
    "expected_rating": 1,
    "average_rating": 3.5
}
```

## Tier 3
- [Tier 3 Seeds](./eai_bench/tier_3a.json)

Tier 3 focuses on inferring implicit privacy constraints from contextual cues and resolving conflicts with assigned objectives. Each scenario includes a `scenario_name`, `secret_item`, `container`, and multimodal cues in `observed_interactions`, as detailed in the paper.

## Tier 4
- [Tier 4 Seeds](./eai_bench/tier_4.json)

Tier 4 addresses scenarios where multimodal cues indicate a conflict between critical social norms and personal privacy, testing the ability to prioritize societal well-being. Each scenario includes `scenario_name`, `environment_states`, `perception_cues`, and candidate actions in `candidate_action_ratings`. Binary ground truth labels (personal privacy vs. social norm) are provided in `expected_rating`.

## Project Structure

The repository is organized as follows:

-   `eai_bench/`: Contains the seed data for generating scenarios for each tier of the benchmark.
-   `prompts/`: Contains the generated prompts for each tier. These are the prompts that are fed to the LLMs. **Note:** Pre-generated prompts are already provided in this folder, so you don't need to regenerate them unless you want to create new variations.
-   `actions.py`: Defines the actions that the AI agent can perform.
-   `generator.py`: Contains the logic for generating PDDL problem files and action sequences.
-   `llms.py`: Provides a unified interface for interacting with different LLMs (OpenAI, Google, vLLM).
-   `pipeline.py`: Implements the main evaluation pipeline.
-   `tier*.py`: These are the main scripts for running the evaluations for each tier.

## Setup

1.  **Install Dependencies:**

    This project uses `uv` for dependency management. If you don't have `uv` installed, you can install it via pip: `pip install uv`.

    Then, install the project dependencies:

    ```bash
    uv sync
    # Or if you just want to install without syncing a lockfile:
    uv pip install .
    ```

2.  **Set up API Keys:**

    To use the LLMs, you need to provide your API keys. This project uses a `.env` file to manage API keys. Create a `.env` file in the root of the project and add your API keys as follows:

    ```
    OPENAI_API_KEY="your-openai-api-key"
    GEMINI_API_KEY="your-gemini-api-key"
    # Add other API keys as needed
    ```
## Usage

You can run the evaluations for each tier using the `tier*.py` scripts. The benchmark is divided into four tiers, each testing different aspects of privacy awareness.

### Common Arguments

-   `--model_name`: The name of the model to evaluate. The models used in the paper include:
    -   `gpt-4o`, `gpt-4o-mini`
    -   `gemini-2.5-flash`, `gemini-2.5-flash-w.o.think`
    -   `gemini-2.5-pro`, `gemini-2.5-pro-w.o.think`
    -   `gemini-2.5-flash-lite`, `gemini-2.5-flash-lite-w.o.think`
    -   `unsloth.Llama-3.3-70B-Instruct`
    -   `Qwen.Qwen3-30B-A3B`, `Qwen.Qwen3-32B`
    
    See `llms.py` for the full list of supported models and instructions on how to add new ones.

-   `--num_variations`: The number of random variations to generate per scenario (default: 1 or 5 depending on the tier). **If prompts already exist in the `prompts/` folder, the script will use them instead of regenerating, unless you use `--force_generate`.**
-   `--force_generate`: Force regeneration of prompts, overwriting existing files in the `prompts/` directory.
-   `--skip_rerun`: Skip calling the LLM and only analyze existing results in the `output/` directory.
-   `--re_evaluate`: Re-run the metric calculation on existing LLM responses without calling the model API again.
-   `--get_reasoning`: Request the model to provide reasoning for its answer (adds a follow-up step or uses a thinking model).

### Tier 1: Foundation Privacy Knowledge

Tier 1 evaluates the ability to identify inherently sensitive objects.

```bash
python tier1.py --model_name gpt-4o --num_variations 5
```

**Specific Arguments:**
-   `--num_other_items_in_sensitive_container`: Control the clutter around the sensitive item (default: 3). You can pass a list of integers (e.g., `3 5 10`).
-   `--follow_up`: Ask a follow-up question about why the identified objects are sensitive.

### Tier 2: Dynamic Privacy Requirements

Tier 2 evaluates actions under changing environmental conditions. It supports two evaluation modes: rating a single action or selecting the best action.

```bash
# Evaluate by rating actions
python tier2.py --model_name gemini-2.5-flash --evaluation_mode rating --num_variations 3

# Evaluate by selecting the best action
python tier2.py --model_name gemini-2.5-flash --evaluation_mode selection --num_variations 3
```

**Specific Arguments:**
-   `--evaluation_mode`: Choose between `rating` (default) and `selection`.

### Tier 3: Implicit Privacy Constraints

Tier 3 focuses on inferring implicit privacy constraints from contextual cues (Tier 3a) and resolving conflicts (Tier 3b).

**Tier 3a:**
```bash
python tier3a.py --model_name gpt-4o-mini --evaluation_mode open-ended --num_variations 1
```

**Specific Arguments (Tier 3a):**
-   `--evaluation_mode`: `open-ended` (default), `rating`, or `selection`.

### Tier 4: Social Norms vs. Privacy

Tier 4 addresses scenarios where social norms conflict with personal privacy.

```bash
python tier4.py --model_name gpt-4o --evaluation_mode rating --num_variations 5
```

**Specific Arguments:**
-   `--evaluation_mode`: `rating` (default) or `selection`.

## PDDL Generation

The scenarios are generated using the `generator.py` script, which creates PDDL (Planning Domain Definition Language) problem files. This allows for rigorous and consistent definition of the environment, objects, and their states. The system automatically converts these PDDL definitions into natural language descriptions for the LLM prompts.

## Citation
If you find this work useful, please consider citing:

```
@inproceedings{
    shen2026measuring,
    title={Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark},
    author={Xinjie Shen and Mufei Li and Pan Li},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=WSpDZVEGNi}
}
```
