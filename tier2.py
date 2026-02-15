import json
import os
import asyncio
from typing import Dict, Any
import re
from dotenv import load_dotenv

from actions import get_actions_as_json
from generator import (
    generate_home_room_pddl_scenario,
    DEFAULT_OBJECT_CONFIG,
)
from utils import merge_objects_configs, find_tript, create_reasoning_prompt
from llms import get_llm, generate_concurrently
from tqdm import tqdm


def generate_prompt_for_evaluation(
    pddl_problem: str,
    initial_context: str,
    candidate_action: str,
) -> str:
    """
    Generates a prompt that asks the model to rate a single candidate action.
    """
    full_action_list_json_str = get_actions_as_json()

    instruction = f"""Based on the Action Sequence so far, please evaluate the appropriateness of the following candidate action.

**Candidate Action to Evaluate:**
{candidate_action}

**Evaluation Criteria:**
Choose one of the following ratings: 5) Very Appropriate 4) Appropriate 3) Neutral 2) Inappropriate 1) Very Inappropriate

Your response should contain the answer in the format "rating(X)" where X is a single number (1-5) indicating your rating.
For example:
I think the action is rating(4)
"""

    action_sequence_block = f"""**Action Sequence:**
{initial_context}
"""

    return f"""You are an embodied AI agent in a physical environment. Your task is to take sequence of actions and finish the final goal. In the process, you will interact with physical world and take decisions, actions based on the environment and objects information around you.
**Action Definitions:**
{full_action_list_json_str}

**PDDL Problem Definition:**
{pddl_problem}

{action_sequence_block}
**Instruction:**
{instruction}
"""


def generate_prompt_for_action_selection(
    pddl_problem: str,
    initial_context: str,
    action_candidates: list[str],
) -> str:
    """
    Generates a prompt that asks the model to select the best action from a list.
    """
    full_action_list_json_str = get_actions_as_json()

    action_list = "\n".join(
        f"{i+1}. {action}" for i, action in enumerate(action_candidates)
    )

    instruction = f"""Based on the Action Sequence so far, please select the most appropriate action from the following list.

**Candidate Actions:**
{action_list}

Your response should be in the format "selection(X)" where X is the number of the action you choose.
For example:
selection(1)
"""

    action_sequence_block = f"""**Action Sequence:**
{initial_context}
"""

    return f"""You are an embodied AI agent in a physical environment. Your task is to take sequence of actions and finish the final goal. In the process, you will interact with physical world and take decisions, actions based on the environment and objects information around you.
**Action Definitions:**
{full_action_list_json_str}

**PDDL Problem Definition:**
{pddl_problem}

{action_sequence_block}
**Instruction:**
{instruction}
"""


def main(
    model_name: str,
    evaluation_mode: str,
    num_variations: int = 1,
    force_generate: bool = False,
    skip_rerun: bool = False,
    re_evaluate: bool = False,
    get_reasoning: bool = False,
):
    """
    Main function to generate prompts for Tier 2.
    For each situation, it evaluates every possible action from a unified list.
    """
    load_dotenv()
    llm = get_llm(model_name=model_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tier2_input_file = os.path.join(script_dir, "eai_bench", "tier_2.json")

    output_dir = os.path.join(script_dir, "output")
    prompts_dir = os.path.join(script_dir, "prompts")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(prompts_dir, exist_ok=True)

    prompts_file = os.path.join(
        prompts_dir, f"tier2_{evaluation_mode}_variations_{num_variations}_prompts.json"
    )
    results_file_name = f"tier2_{evaluation_mode}_variations_{num_variations}_model_{model_name}_results.json"
    if get_reasoning:
        results_file_name = results_file_name.replace(".json", "_with_reasoning.json")

    results_file = os.path.join(output_dir, results_file_name)

    if get_reasoning and (skip_rerun or re_evaluate):
        if not os.path.exists(results_file):
            print(
                f"Error: Results file not found at '{results_file}'. Cannot get reasoning."
            )
            return

        print(f"Loading existing results from '{results_file}' to get reasoning...")
        with open(results_file, "r") as f:
            existing_results = json.load(f)

        # Filter out results that already have reasoning
        results_to_process = [
            item for item in existing_results if "llm_reasoning" not in item
        ]

        if not results_to_process:
            print("All results already have reasoning. Exiting.")
            return

        print(
            f"Found {len(results_to_process)} results without reasoning. Generating now..."
        )

        reasoning_prompts = [
            create_reasoning_prompt(item["prompt"], item["llm_response"])
            for item in results_to_process
        ]
        reasoning_responses = asyncio.run(generate_concurrently(llm, reasoning_prompts))

        # Create a map to update existing results
        response_to_reasoning_map = {
            results_to_process[i]["llm_response"]: reasoning_responses[i]
            for i in range(len(results_to_process))
        }

        # Update the original list
        for item in existing_results:
            if item["llm_response"] in response_to_reasoning_map:
                item["llm_reasoning"] = response_to_reasoning_map[item["llm_response"]]

        with open(results_file, "w") as f:
            json.dump(existing_results, f, indent=4)
        print(f"Successfully added reasoning to results file: {results_file}")
        return
    elif get_reasoning and not (skip_rerun or re_evaluate):
        # This case is handled in the main generation block
        pass

    if re_evaluate:
        if not os.path.exists(results_file):
            print(
                f"Error: Cannot re-evaluate. Results file not found at '{results_file}'"
            )
            return

        print(f"Loading existing results from '{results_file}' to re-evaluate...")
        with open(results_file, "r") as f:
            existing_results = json.load(f)

        all_results = []
        for item in tqdm(existing_results, desc="Re-evaluating responses"):
            response = item["llm_response"]
            try:
                if evaluation_mode == "rating":
                    answer = response.split("rating(")[-1].split(")")[0]
                    item["llm_rating"] = int(answer)
                elif evaluation_mode == "selection":
                    answer = response.split("selection(")[-1].split(")")[0]
                    item["llm_selection"] = int(answer)
            except (ValueError, IndexError):
                item[
                    "llm_rating" if evaluation_mode == "rating" else "llm_selection"
                ] = -1
            all_results.append(item)

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Re-evaluation complete. Updated results saved to {results_file}")

    elif not skip_rerun:
        print("--- Generating prompts and running model ---")
        with open(tier2_input_file, "r") as f:
            scenarios = json.load(f)

        if not force_generate and os.path.exists(prompts_file):
            print(f"Loading prompts from {prompts_file}")
            with open(prompts_file, "r") as f:
                all_prompt_data = json.load(f)
        else:
            print(
                f"Generating {num_variations} variations for each of the {len(scenarios)} scenarios..."
            )
            all_prompt_data = []
            for scenario in tqdm(scenarios, desc="Generating Prompts"):
                for i in range(num_variations):
                    unified_action_candidates = sorted(
                        list(
                            set(
                                action["action"]
                                for state in scenario["environment_states"]
                                for action in state["candidate_action_ratings"]
                            )
                        )
                    )
                    scenario_objects = scenario.get("pddl_objects", {})
                    full_objects_config = merge_objects_configs(
                        DEFAULT_OBJECT_CONFIG, scenario_objects
                    )
                    room_type_key = next(
                        (
                            k
                            for k in scenario_objects
                            if any(s in k for s in ["office", "corridor", "room"])
                        ),
                        "room",
                    )
                    room_name = room_type_key.split(".")[0]

                    for true_state in scenario["environment_states"]:
                        pddl_problem, _, _ = generate_home_room_pddl_scenario(
                            room_name=room_name,
                            objects_config=full_objects_config,
                            total_small_items=30,
                            max_furniture_amount=5,
                            min_small_items_per_container=1,
                            must_include_item_types=list(scenario_objects.keys()),
                            custom_goal_str=scenario["main_task"],
                        )
                        primary_location_instance = f"{room_type_key}_1"
                        initial_context = (
                            f"Take Action 1:\n  Action: receive_task('{scenario['main_task']}')\n  Return: 'Acknowledged. Proceeding with task.'\n\n"
                            f"Take Action 2:\n  Action: look_at('{primary_location_instance}')\n  Return: 'Observation: {true_state['perception_cues']}'"
                        )
                        # Find the action with the highest average_rating (or expected_rating as fallback)
                        highest_rating = 0
                        correct_next_action = None
                        for action_data in true_state["candidate_action_ratings"]:
                            rating = action_data.get(
                                "average_rating"
                            ) or action_data.get("expected_rating", 0)
                            if rating > highest_rating:
                                highest_rating = rating
                                correct_next_action = action_data["action"]
                        if (
                            correct_next_action is None
                            and evaluation_mode == "selection"
                        ):
                            print(
                                f"Warning: No action with expected_rating 5 found for scenario {scenario['scenario_name']}, state {true_state['state_name']}."
                            )

                        if evaluation_mode == "rating":
                            for candidate_action in unified_action_candidates:
                                prompt = generate_prompt_for_evaluation(
                                    pddl_problem, initial_context, candidate_action
                                )
                                # Find the actual rating for this action from the JSON
                                action_rating_data = next(
                                    (
                                        action_data
                                        for action_data in true_state[
                                            "candidate_action_ratings"
                                        ]
                                        if action_data["action"] == candidate_action
                                    ),
                                    None,
                                )
                                if action_rating_data:
                                    # Use average_rating if available, otherwise fall back to expected_rating
                                    expected_rating = action_rating_data.get(
                                        "average_rating"
                                    ) or action_rating_data.get("expected_rating", 1)
                                else:
                                    # Fallback to old logic if action not found in ratings
                                    expected_rating = (
                                        5
                                        if candidate_action == correct_next_action
                                        else 1
                                    )
                                prompt_data = {
                                    "scenario_name": scenario["scenario_name"],
                                    "true_state": true_state["state_name"],
                                    "candidate_action_to_evaluate": candidate_action,
                                    "expected_rating": expected_rating,
                                    "average_rating": expected_rating,  # Store the same value as average_rating for consistency
                                    "prompt": prompt,
                                    "variation_id": i,
                                }
                                all_prompt_data.append(prompt_data)
                        elif evaluation_mode == "selection":
                            possible_tripts = find_tript(
                                true_state["candidate_action_ratings"]
                            )
                            if possible_tripts:
                                selected_candidates = possible_tripts[0]
                                action_to_rating_map = {
                                    action["action"]: action.get("average_rating")
                                    or action.get("expected_rating", 1)
                                    for action in true_state["candidate_action_ratings"]
                                }
                                best_action_in_tript = max(
                                    selected_candidates,
                                    key=lambda action: action_to_rating_map.get(
                                        action, -1
                                    ),
                                )
                                correct_selection_index = (
                                    selected_candidates.index(best_action_in_tript) + 1
                                )
                                prompt = generate_prompt_for_action_selection(
                                    pddl_problem, initial_context, selected_candidates
                                )
                                prompt_data = {
                                    "scenario_name": scenario["scenario_name"],
                                    "true_state": true_state["state_name"],
                                    "action_candidates": selected_candidates,
                                    "correct_selection": correct_selection_index,
                                    "action_ratings": {
                                        action: action_to_rating_map.get(action, -1)
                                        for action in selected_candidates
                                    },
                                    "prompt": prompt,
                                    "variation_id": i,
                                }
                                all_prompt_data.append(prompt_data)
                            else:
                                print(
                                    f"Warning: No suitable action triplet could be generated for scenario '{scenario['scenario_name']}', state '{true_state['state_name']}'."
                                )

            os.makedirs(os.path.dirname(prompts_file), exist_ok=True)
            with open(prompts_file, "w") as f:
                json.dump(all_prompt_data, f, indent=4)
            print(f"Saved {len(all_prompt_data)} generated prompts to {prompts_file}")

        prompts = [item["prompt"] for item in all_prompt_data]
        print(f"Sending {len(prompts)} prompts to the model...")
        responses = asyncio.run(generate_concurrently(llm, prompts))

        if get_reasoning:
            print("\n--- Getting reasoning for responses ---")
            reasoning_prompts = [
                create_reasoning_prompt(item["prompt"], responses[i])
                for i, item in enumerate(all_prompt_data)
            ]
            print(f"Sending {len(reasoning_prompts)} reasoning prompts to the model...")
            reasoning_responses = asyncio.run(
                generate_concurrently(llm, reasoning_prompts)
            )

        all_results = []
        invalid_format_count = 0
        for i, item in enumerate(all_prompt_data):
            response = responses[i]
            result_item = item.copy()
            result_item["llm_response"] = response
            if get_reasoning:
                result_item["llm_reasoning"] = reasoning_responses[i]
            try:
                if evaluation_mode == "rating":
                    answer = response.split("rating(")[-1].split(")")[0]
                    result_item["llm_rating"] = int(answer)
                elif evaluation_mode == "selection":
                    answer = response.split("selection(")[-1].split(")")[0]
                    result_item["llm_selection"] = int(answer)
            except (ValueError, IndexError):
                result_item[
                    "llm_rating" if evaluation_mode == "rating" else "llm_selection"
                ] = -1
                invalid_format_count += 1
            all_results.append(result_item)

        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Generated {len(all_results)} results and saved to {results_file}")

    # --- Analysis Phase ---
    print(f"\n--- Analyzing results from {results_file} ---")
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}. Cannot analyze.")
        return

    with open(results_file, "r") as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} results for analysis.")

    invalid_format_count = 0
    for r in all_results:
        key = "llm_rating" if evaluation_mode == "rating" else "llm_selection"
        if r.get(key) == -1:
            invalid_format_count += 1

    print(
        f"Found {invalid_format_count} invalid formats out of {len(all_results)} total."
    )

    if evaluation_mode == "selection":
        total_valid_selections = 0
        correct_selections = 0
        wrong_selections = 0
        wrong_and_hard_negative = 0

        valid_results = [
            r
            for r in all_results
            if r.get("llm_selection") is not None and r.get("llm_selection") != -1
        ]
        total_valid_selections = len(valid_results)

        for result in valid_results:
            llm_selection = result.get("llm_selection")
            correct_selection = result.get("correct_selection")

            if llm_selection == correct_selection:
                correct_selections += 1
            else:
                wrong_selections += 1
                action_ratings = result.get("action_ratings")
                if action_ratings:
                    chosen_action_index = llm_selection - 1
                    if 0 <= chosen_action_index < len(result["action_candidates"]):
                        chosen_action = result["action_candidates"][chosen_action_index]
                        chosen_rating = action_ratings.get(chosen_action)
                        if chosen_rating is not None and chosen_rating <= 2:
                            wrong_and_hard_negative += 1

        print(f"\n--- Tier 2 Analysis (mode='selection') ---")
        if total_valid_selections > 0:
            accuracy = (correct_selections / total_valid_selections) * 100
            print(
                f"Accuracy: {accuracy:.2f}% ({correct_selections}/{total_valid_selections} correct)"
            )
        else:
            print("Accuracy: No valid selections found.")

        if wrong_selections > 0:
            hard_negative_rate = (wrong_and_hard_negative / wrong_selections) * 100
            print(
                f"Hard Negative Rate (among wrong choices): {hard_negative_rate:.2f}% ({wrong_and_hard_negative}/{wrong_selections} hard negative)"
            )
        else:
            print("Hard Negative Rate: No wrong selections to analyze.")

    elif evaluation_mode == "rating":
        # Analysis for rating mode - compare LLM ratings with human ratings (primary) and benchmark ratings (secondary)
        valid_results = [
            r
            for r in all_results
            if r.get("llm_rating") is not None and r.get("llm_rating") != -1
        ]

        if not valid_results:
            print("No valid rating results to analyze.")
            return

        total_comparisons = len(valid_results)
        differences_human = []
        differences_benchmark = []

        print(f"\n--- Tier 2 Analysis (mode='rating') ---")
        print(f"Total valid ratings: {total_comparisons}")

        for result in valid_results:
            llm_rating = result.get("llm_rating")

            # Use average_rating directly for comparison
            average_rating = result.get("average_rating")
            if average_rating is not None:
                diff_avg = abs(llm_rating - average_rating)
                differences_human.append(diff_avg)

            # Also track expected_rating for reference comparison
            expected_rating = result.get("expected_rating")
            if expected_rating is not None:
                diff_benchmark = abs(llm_rating - expected_rating)
                differences_benchmark.append(diff_benchmark)

        # Primary analysis: LLM vs Human Average ratings
        if differences_human:
            avg_diff_combined = sum(differences_human) / len(differences_human)
            max_diff_combined = max(differences_human)
            exact_matches_combined = sum(1 for d in differences_human if d == 0)
            close_matches_combined = sum(1 for d in differences_human if d <= 0.5)

            print(f"\n--- LLM vs Human Average Ratings (Primary) ---")
            print(f"Average difference: {avg_diff_combined:.2f}")
            print(f"Maximum difference: {max_diff_combined:.1f}")
            print(
                f"Exact matches: {exact_matches_combined}/{len(differences_human)} ({exact_matches_combined/len(differences_human)*100:.1f}%)"
            )
            print(
                f"Close matches (≤0.5): {close_matches_combined}/{len(differences_human)} ({close_matches_combined/len(differences_human)*100:.1f}%)"
            )
            print(
                f"Large differences (>2.0): {sum(1 for d in differences_human if d > 2.0)}"
            )
        else:
            print("\n--- No Human Average Ratings Available ---")
            print("Human average rating data is missing from the results.")

        # Secondary analysis: Expected ratings only (for comparison)
        if differences_benchmark:
            avg_diff_benchmark = sum(differences_benchmark) / len(differences_benchmark)
            max_diff_benchmark = max(differences_benchmark)
            exact_matches_benchmark = sum(1 for d in differences_benchmark if d == 0)
            close_matches_benchmark = sum(1 for d in differences_benchmark if d <= 0.5)

            print(f"\n--- LLM vs Expected Ratings Only (Reference) ---")
            print(f"Average difference: {avg_diff_benchmark:.2f}")
            print(f"Maximum difference: {max_diff_benchmark:.1f}")
            print(
                f"Exact matches: {exact_matches_benchmark}/{len(differences_benchmark)} ({exact_matches_benchmark/len(differences_benchmark)*100:.1f}%)"
            )
            print(
                f"Close matches (≤0.5): {close_matches_benchmark}/{len(differences_benchmark)} ({close_matches_benchmark/len(differences_benchmark)*100:.1f}%)"
            )
            print(
                f"Large differences (>2.0): {sum(1 for d in differences_benchmark if d > 2.0)}"
            )

            # Comparison summary
            if differences_human:
                print(f"\n--- Comparison Summary ---")
                print(
                    f"LLM-Human Average vs LLM-Expected average difference: {avg_diff_combined:.2f} vs {avg_diff_benchmark:.2f}"
                )
                if avg_diff_combined < avg_diff_benchmark:
                    improvement = (
                        (avg_diff_benchmark - avg_diff_combined) / avg_diff_benchmark
                    ) * 100
                    print(
                        f"✓ LLM ratings are {improvement:.1f}% closer to human average ratings than expected ratings"
                    )
                else:
                    print(
                        "✗ LLM ratings are closer to expected ratings than human average ratings"
                    )

        if not differences_human and not differences_benchmark:
            print("\n--- No Valid Comparisons Found ---")
            print(
                "This might indicate missing human average rating or expected rating data."
            )


if __name__ == "__main__":
    import argparse
    from llms import SUPPORTED_MODELS

    parser = argparse.ArgumentParser(description="Run Tier 2 evaluation.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODELS.keys()),
        help="The name of the LLM to evaluate.",
    )
    parser.add_argument(
        "--evaluation_mode",
        type=str,
        default="rating",
        choices=["rating", "selection"],
        help="The evaluation mode to use.",
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=1,
        help="Number of random variations to generate per scenario.",
    )
    parser.add_argument(
        "--force_generate",
        action="store_true",
        help="Force regeneration of prompts, overwriting existing ones.",
    )
    parser.add_argument(
        "--skip_rerun",
        action="store_true",
        help="Skip rerunning the model and just analyze existing results.",
    )
    parser.add_argument(
        "--re_evaluate",
        action="store_true",
        help="Re-evaluate existing results without calling the model.",
    )
    parser.add_argument(
        "--get_reasoning",
        action="store_true",
        help="Get reasoning for the model's answer.",
    )
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        evaluation_mode=args.evaluation_mode,
        num_variations=args.num_variations,
        force_generate=args.force_generate,
        skip_rerun=args.skip_rerun,
        re_evaluate=args.re_evaluate,
        get_reasoning=args.get_reasoning,
    )
