# EAPrivacy - Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark

[![arXiv](https://img.shields.io/badge/📄arXiv-2510.02356-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e)](https://arxiv.org/abs/2510.02356)

![Theme Image](./assets/theme_pic.png)

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

## Release Plan

The code for scenario generation and LLM evaluation is being refactored and will be released after paper acceptance. This codebase is maintained by the [G-COM group](https://github.com/Graph-COM/EAPrivacy).

## Citation
If you find this work useful, please consider citing:

```
@misc{shen2025measuringphysicalworldprivacyawareness,
      title={Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark}, 
      author={Xinjie Shen and Mufei Li and Pan Li},
      year={2025},
      eprint={2510.02356},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2510.02356}, 
}
```