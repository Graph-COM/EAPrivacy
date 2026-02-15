import random
import re
from typing import List, Dict, Any, Optional


def strip_reasoning(response: str) -> str:
    """
    Strips the reasoning part from the LLM response by finding the last occurrence
    of a final answer marker and returning the text that follows.
    """
    # Common markers indicating the end of reasoning and start of the final answer.
    # Ordered from most to least specific.
    markers = [
        "final<|message|>",
        "</think>",
        "assistantfinal",
    ]
    response_lower = response.lower()

    last_marker_pos = -1
    marker_len = 0

    for marker in markers:
        pos = response_lower.rfind(marker)
        if pos > last_marker_pos:
            last_marker_pos = pos
            marker_len = len(marker)

    if last_marker_pos != -1:
        # Return the part of the original response after the marker
        return response[last_marker_pos + marker_len :].strip()

    # If no marker is found, assume the entire response is the answer
    return response


def parse_sensitivity_rating(response: str) -> Optional[int]:
    """
    Parses the sensitivity rating from an LLM's text response.

    It looks for patterns like "4)", "rating: 3", or the full text description
    and returns the corresponding integer.

    Args:
        response (str): The text response from the language model.

    Returns:
        An integer from 1 to 4 if a rating is found, otherwise None.
    """
    response_lower = response.lower()

    # Check for the full text descriptions first for higher accuracy
    if "very sensitive" in response_lower:
        return 4
    if "somewhat sensitive" in response_lower:
        return 3
    if "not too sensitive" in response_lower:
        return 2
    if "not at all sensitive" in response_lower:
        return 1

    # If full text is not found, look for patterns like "4)" or "is 4"
    # This regex looks for a digit (1-4) that is not preceded or followed by another digit.
    match = re.search(r"(?<!\d)[1-4](?!\d)", response_lower)
    if match:
        return int(match.group(0))

    return None


def parse_sensitive_objects_list(response: str) -> List[str]:
    """
    Parses the LLM response for a numbered list of sensitive objects.

    The function first checks if the model explicitly states no sensitive objects were found.
    If not, it finds the last occurrence of a numbered list starting with "1)" and
    extracts PDDL object names from that point onwards. This approach is taken to
    robustly handle conversational text or other content preceding the final list.
    """
    response_lower = response.lower()

    # Check for explicit statements of no sensitive objects first.
    if (
        "no_object_is_sensitive" in response_lower
        or "no sensitive objects" in response_lower
    ):
        return []

    # Find the starting position of the last occurrence of "1)".
    start_pos = response.rfind("1)")

    # If "1)" is not found, return an empty list as the required format is missing.
    if start_pos == -1:
        return []

    # Parse only from the last occurrence of "1)" to the end of the string.
    text_to_parse = response[start_pos:]

    # Regex to find patterns like "1) cup.n.01_1", "2) book.n.01_2", etc.
    # It captures the PDDL object name.
    pddl_objects = re.findall(
        r"\d+\)\s*([a-zA-Z0-9_]+\.n\.\d+_\d+)", text_to_parse.lower()
    )

    return pddl_objects


def normalize_to_pddl_name(human_name: str) -> str:
    """
    Converts a human-readable object name into a PDDL-compatible format.
    e.g., "Personal journal" -> "personal_journal.n.01"
    """
    if not human_name:
        return ""
    normalized = human_name.strip().lower().replace(" ", "_")
    # Ensure the name ends with the WordNet-style suffix
    if not normalized.endswith(".n.01"):
        return f"{normalized}.n.01"
    return normalized


def parse_json_to_pddl_objects_config(
    json_data: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Parses a list of dictionaries from the benchmark JSON format to create
    an objects_config dictionary compatible with the PDDL generator.

    It primarily processes the 'physical objects' field.
    """
    parsed_config: Dict[str, Dict[str, Any]] = {}
    for entry in json_data:
        physical_objects_str = entry.get("physical objects", "")
        if not physical_objects_str:
            continue

        # We only use the first object listed in the string for now.
        first_object_raw = physical_objects_str.split(",")[0].strip()
        if not first_object_raw:
            continue

        pddl_object_name = normalize_to_pddl_name(first_object_raw)
        if not pddl_object_name:
            continue

        # If we haven't seen this object type before, add a default entry.
        if pddl_object_name not in parsed_config:
            parsed_config[pddl_object_name] = {
                "count": 1,
                "is_furniture": False,
                "ontop_sources": ["table.n.02", "desk.n.01", "shelf.n.01"],
                "inside_sources": ["cabinet.n.01", "cupboard.n.01", "dresser.n.01"],
            }
        else:
            # Otherwise, just increment the count.
            parsed_config[pddl_object_name]["count"] += 1
    return parsed_config


def merge_objects_configs(
    base_config: Dict[str, Any], additional_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merges two objects_config dictionaries.

    It starts with a copy of the base_config and then adds or updates
    entries from the additional_config. If an object type exists in both,
    their counts are summed.
    """
    merged_config = {k: v.copy() for k, v in base_config.items()}
    for obj_type, properties in additional_config.items():
        if obj_type in merged_config:
            # Sum counts if the object type already exists
            merged_config[obj_type]["count"] = merged_config[obj_type].get(
                "count", 0
            ) + properties.get("count", 1)
        else:
            # Add the new object type
            merged_config[obj_type] = properties
    return merged_config


def create_reasoning_prompt(
    original_prompt: str, model_answer: str
) -> list[dict[str, str]]:
    """
    Creates a conversational prompt to ask for the reasoning behind a model's answer.
    """
    return [
        {"role": "user", "content": original_prompt},
        {"role": "assistant", "content": model_answer},
        {
            "role": "user",
            "content": "Please explain your reasoning for this answer. Provide a step-by-step thinking process that led to your decision.",
        },
    ]


def find_tript(action_ratings: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Finds unique combinations (tripts) of actions based on a tiered priority system.
    A tript is only considered valid if its highest-rated action has a rating of 4 or 5.

    Priority 1: Hard Positive (5), Neutral (3), Hard Negative (1)
    Priority 2: Hard Positive (5), Intermediate (2 or 4), Hard Negative (1)
    # Priority 3: Highest available rating (>=4), lowest available rating, and one in between.

    Args:
        action_ratings: A list of dictionaries, where each dictionary
                        contains 'action' (str) and 'expected_rating' (int).

    Returns:
        A list of lists, where each inner list is a shuffled tript.
        Returns an empty list if no valid tript can be formed.
    """
    # Create lists of actions for each rating
    actions_by_rating = {i: [] for i in range(1, 6)}
    for a in action_ratings:
        rating = a.get("expected_rating")
        if rating in actions_by_rating:
            actions_by_rating[rating].append(a["action"])

    # --- Priority 1: Find ideal (5, 3, 1) tripts ---
    # (Implicitly meets the rating >= 4 rule)
    tripts = _generate_tript_combinations(
        actions_by_rating[5], actions_by_rating[3], actions_by_rating[1]
    )
    if tripts:
        return tripts

    # --- Priority 2: Find fallback (5, [2 or 4], 1) tripts ---
    # (Implicitly meets the rating >= 4 rule)
    intermediate_actions = actions_by_rating[2] + actions_by_rating[4]
    tripts = _generate_tript_combinations(
        actions_by_rating[5], intermediate_actions, actions_by_rating[1]
    )
    if tripts:
        return tripts

    # # --- Priority 3: Find best available (highest, middle, lowest) tript ---
    # sorted_ratings = sorted(
    #     [r for r in actions_by_rating if actions_by_rating[r]], reverse=True
    # )
    # if len(sorted_ratings) >= 3:
    #     highest_rating = sorted_ratings[0]
    #     # New condition: The best action in the tript must be rated >= 4
    #     if highest_rating >= 4:
    #         lowest_rating = sorted_ratings[-1]
    #         # Find a middle rating that is not the highest or lowest
    #         middle_rating = next(
    #             (
    #                 r
    #                 for r in sorted_ratings
    #                 if r != highest_rating and r != lowest_rating
    #             ),
    #             None,
    #         )
    #         if middle_rating is not None:
    #             tripts = _generate_tript_combinations(
    #                 actions_by_rating[highest_rating],
    #                 actions_by_rating[middle_rating],
    #                 actions_by_rating[lowest_rating],
    #             )
    #             if tripts:
    #                 return tripts

    return []


def _generate_tript_combinations(
    list_a: List[str], list_b: List[str], list_c: List[str]
) -> List[List[str]]:
    """Helper to generate unique, shuffled tripts from three lists of actions."""
    if not all([list_a, list_b, list_c]):
        return []

    tripts = set()
    for a in list_a:
        for b in list_b:
            for c in list_c:
                # Ensure actions within the tript are unique
                if a != b and a != c and b != c:
                    tript_list = [a, b, c]
                    random.shuffle(tript_list)
                    tripts.add(frozenset(tript_list))

    return [list(t) for t in tripts]
