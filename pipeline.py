import os
import json
import asyncio
import numpy as np
from typing import Callable, Dict, Any, List

from llms import LLM_General, generate_concurrently


def run_evaluation_pipeline(
    llm: LLM_General,
    run_name: str,
    prompts_file: str,
    results_file: str,
    num_responses: int,
    prompt_generator: Callable[[Dict[str, Any]], str],
    response_evaluator: Callable[[str, Dict[str, Any]], int],
    prompt_input_file: str = None,
    generate_prompts: bool = False,
    skip_rerun: bool = False,
):
    """
    Runs the full pipeline for a given set of prompts using concurrent generation.
    """
    print(f"\n{'='*20}\n--- Starting Pipeline for: {run_name} ---\n{'='*20}")

    # --- Step 1: Generate or Load Prompts ---
    if generate_prompts and prompt_input_file:
        print(f"Generating new prompts and saving to '{prompts_file}'...")
        with open(prompt_input_file, "r") as f:
            json_input_data = json.load(f)

        all_prompts = [
            {"prompt_text": prompt_generator(entry), "input_data": entry}
            for entry in json_input_data
        ]

        os.makedirs(os.path.dirname(prompts_file), exist_ok=True)
        with open(prompts_file, "w") as output_file:
            json.dump(all_prompts, output_file, indent=2)
        print(f"Successfully generated and saved {len(all_prompts)} prompts.")
    else:
        print(f"Loading prompts from '{prompts_file}'...")
        with open(prompts_file, "r") as f:
            all_prompts = json.load(f)

    # --- Step 2: Generate LLM Responses ---
    results = []
    if skip_rerun and os.path.exists(results_file):
        print(
            f"Skipping generation and loading existing results from '{results_file}'..."
        )
        with open(results_file, "r") as f:
            results = json.load(f)
    elif os.path.exists(results_file) and not generate_prompts:
        print(f"Loading existing results from '{results_file}'...")
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        print(
            f"Generating {num_responses} responses for each of the {len(all_prompts)} prompts..."
        )

        # Create a flat list of all prompts to be generated
        prompts_to_generate = [
            item["prompt_text"] for item in all_prompts for _ in range(num_responses)
        ]

        # Generate all responses concurrently
        all_generated_responses = asyncio.run(
            generate_concurrently(llm, prompts_to_generate)
        )

        # Re-structure the results back to group responses by prompt
        response_idx = 0
        for prompt_item in all_prompts:
            responses = all_generated_responses[
                response_idx : response_idx + num_responses
            ]
            results.append({"prompt_item": prompt_item, "responses": responses})
            response_idx += num_responses

        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n--- LLM generation complete. Results saved to '{results_file}'. ---")

    # --- Step 3: Evaluate Results ---
    print(f"\n--- Evaluating results from '{results_file}' ---")
    all_ratings = []
    for result in results:
        prompt_item = result["prompt_item"]
        ratings = [
            response_evaluator(res, prompt_item["input_data"])
            for res in result.get("responses", [])
        ]
        valid_ratings = [r for r in ratings if r is not None]
        if valid_ratings:
            all_ratings.extend(valid_ratings)

    if all_ratings:
        average_rating = np.mean(all_ratings)
        print(f"\nEvaluation Summary for {run_name}:")
        print(f"  - Total valid ratings: {len(all_ratings)}")
        print(f"  - Average Rating: {average_rating:.2f}")
    else:
        print("No valid ratings were generated.")

    return all_ratings
