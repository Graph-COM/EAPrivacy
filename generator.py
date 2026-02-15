import random
from collections import defaultdict
from actions import ACTIONS, LOOK_AT_CASES

# Define a Default Object Configuration for easy use
# Note: Counts here represent *desired* counts. The actual number of small items
# and furniture will be capped by `max_small_items` and `max_furniture_amount` respectively.
DEFAULT_OBJECT_CONFIG = {
    # --- Furniture (Suitable Locations/Containers) ---
    # General Home
    "table.n.02": {
        "count": random.randint(1, 2),
        "is_furniture": True,
        "is_container": False,
    },
    "shelf.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": False,
    },
    "desk.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": False,
    },
    "cupboard.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": True,
    },
    # Living Room specific
    "sofa.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": False,
    },
    "coffee_table.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": False,
    },
    "tv_stand.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": False,
    },
    # Kitchen specific
    "cabinet.n.01": {
        "count": random.randint(1, 2),
        "is_furniture": True,
        "is_container": True,
    },
    "electric_refrigerator.n.01": {
        "count": 1,
        "is_furniture": True,
        "is_container": True,
    },
    "stove.n.01": {
        "count": 1,
        "is_furniture": True,
        "is_container": False,
    },
    "microwave.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": True,
    },
    "counter_top.n.01": {
        "count": random.randint(1, 2),
        "is_furniture": True,
        "is_container": False,
    },
    # Bedroom specific
    "bed.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": False,
    },
    "dresser.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": True,
    },
    "nightstand.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": False,
    },
    "chair.n.01": {
        "count": random.randint(0, 1),
        "is_furniture": True,
        "is_container": False,
    },
    # --- Small objects (Items to be placed) ---
    "note.n.01": {
        "count": random.randint(2, 4),
        "ontop_sources": [
            "table.n.02",
            "desk.n.01",
            "nightstand.n.01",
            "coffee_table.n.01",
        ],
    },
    "cd.n.01": {
        "count": random.randint(1, 3),
        "ontop_sources": ["table.n.02", "shelf.n.01", "desk.n.01", "tv_stand.n.01"],
        "inside_sources": ["cabinet.n.01", "cupboard.n.01"],
    },
    "book.n.01": {
        "count": random.randint(3, 5),
        "ontop_sources": [
            "table.n.02",
            "shelf.n.01",
            "desk.n.01",
            "nightstand.n.01",
            "coffee_table.n.01",
        ],
    },
    "pen.n.01": {
        "count": random.randint(1, 2),
        "ontop_sources": ["table.n.02", "desk.n.01", "nightstand.n.01"],
    },
    "laptop.n.01": {
        "count": 1,
        "ontop_sources": [
            "table.n.02",
            "desk.n.01",
            "nightstand.n.01",
            "coffee_table.n.01",
        ],
    },
    "trophy.n.01": {
        "count": 1,
        "ontop_sources": ["table.n.02", "shelf.n.01", "tv_stand.n.01"],
    },
    "remote_control.n.01": {
        "count": random.randint(1, 2),
        "ontop_sources": ["coffee_table.n.01", "tv_stand.n.01", "sofa.n.01"],
    },
    "plant.n.01": {
        "count": random.randint(1, 2),
        "ontop_sources": [
            "table.n.02",
            "shelf.n.01",
            "desk.n.01",
            "nightstand.n.01",
            "coffee_table.n.01",
            "tv_stand.n.01",
            "counter_top.n.01",
        ],
    },
    "cup.n.01": {
        "count": random.randint(2, 4),
        "ontop_sources": ["table.n.02", "counter_top.n.01", "stove.n.01"],
        "inside_sources": ["cupboard.n.01", "cabinet.n.01"],
    },
    "plate.n.01": {
        "count": random.randint(2, 4),
        "ontop_sources": ["table.n.02", "counter_top.n.01"],
        "inside_sources": ["cupboard.n.01", "cabinet.n.01"],
    },
    "bowl.n.01": {
        "count": random.randint(1, 2),
        "ontop_sources": ["table.n.02", "counter_top.n.01"],
        "inside_sources": ["cupboard.n.01", "cabinet.n.01"],
    },
    "teapot.n.01": {
        "count": 1,
        "ontop_sources": ["table.n.02", "counter_top.n.01", "stove.n.01"],
        "inside_sources": ["cabinet.n.01"],
    },
    "tea_bag.n.01": {
        "count": random.randint(2, 5),
        "inside_sources": ["cabinet.n.01", "cupboard.n.01"],
        "ontop_sources": ["table.n.02", "counter_top.n.01"],
    },
    "lemon.n.01": {
        "count": random.randint(1, 3),
        "inside_sources": ["electric_refrigerator.n.01"],
        "ontop_sources": ["table.n.02", "counter_top.n.01"],
    },
    "knife.n.01": {
        "count": random.randint(1, 3),
        "inside_sources": ["cabinet.n.01"],
        "ontop_sources": ["table.n.02", "counter_top.n.01"],
    },
    "food_item.n.01": {
        "count": random.randint(2, 4),
        "inside_sources": [
            "electric_refrigerator.n.01",
            "cabinet.n.01",
            "cupboard.n.01",
        ],
        "ontop_sources": ["table.n.02", "counter_top.n.01"],
    },
    "fork.n.01": {
        "count": random.randint(2, 4),
        "inside_sources": ["cabinet.n.01"],
        "ontop_sources": ["table.n.02", "counter_top.n.01"],
    },
    "spoon.n.01": {
        "count": random.randint(2, 4),
        "inside_sources": ["cabinet.n.01"],
        "ontop_sources": ["table.n.02", "counter_top.n.01"],
    },
    "pan.n.01": {
        "count": random.randint(0, 1),
        "ontop_sources": ["stove.n.01", "counter_top.n.01"],
        "inside_sources": ["cabinet.n.01", "cupboard.n.01"],
    },
    "pillow.n.01": {
        "count": random.randint(1, 3),
        "ontop_sources": ["sofa.n.01", "bed.n.01"],
    },
    "blanket.n.01": {
        "count": random.randint(1, 1),
        "ontop_sources": ["sofa.n.01", "bed.n.01"],
    },
    "clothes.n.01": {
        "count": random.randint(2, 4),
        "inside_sources": ["dresser.n.01"],
        "ontop_sources": ["bed.n.01", "chair.n.01"],
    },
}


def _create_action(action_name, all_object_instances):
    """Helper to create a random action tuple (action_str, result_text)."""
    random_object = random.choice(all_object_instances)
    obj_type = random_object.split(".")[0]

    if "place" in action_name:
        random_surface = random.choice(all_object_instances)
        surface_type = random_surface.split(".")[0]
        action_str = f"{action_name}('{obj_type}', '{surface_type}')"
        result_text = f"Status: {obj_type} is now on {surface_type}."
    elif "grasp" in action_name:
        action_str = f"{action_name}('{obj_type}')"
        result_text = f"Status: Holding {obj_type}."
    elif "open" in action_name or "close" in action_name:
        action_str = f"{action_name}('{obj_type}')"
        result_text = f"StateChange: {obj_type} is now {action_name}."
    else:
        action_str = f"{action_name}('{obj_type}')"
        result_text = f"Action '{action_name}' executed on {obj_type}."
    return {"action": action_str, "result": result_text}


def generate_action_sequence(
    main_object_instance: str,
    combination_text: str,
    all_instances_by_type: dict,
    max_turns=5,
    human_actor: str = None,
    look_at_ratio: float = 0.0,
):
    """
    Generates a randomized sequence of actions with tool-like natural language results.
    """
    # 1. Create the specific, required actions
    object_type = main_object_instance.rsplit("_", 1)[0]
    look_at_result = (
        LOOK_AT_CASES.get(object_type)
        if object_type
        else f"Observation: You see the {combination_text}."
    )
    specific_actions = [
        {"action": f"look_at('{main_object_instance}')", "result": look_at_result}
    ]
    if human_actor:
        specific_actions.append(
            {
                "action": f"talk_to('{human_actor}', '{main_object_instance}')",
                "result": combination_text,
            }
        )

    # 2. Create a pool of all possible object instances for random actions
    all_object_instances = [
        inst for inst_list in all_instances_by_type.values() for inst in inst_list
    ]
    if not all_object_instances:
        return "No objects available to perform actions on."

    # 3. Generate additional random actions
    num_random_actions = max(0, max_turns - len(specific_actions))
    random_actions = []
    generic_action_pool = {
        k: v for k, v in ACTIONS.items() if k not in ["look_at", "talk_to"]
    }

    for _ in range(num_random_actions):
        if random.random() < look_at_ratio:
            random_instance = random.choice(all_object_instances)
            instance_type = random_instance.rsplit("_", 1)[0]
            result_text = LOOK_AT_CASES.get(
                instance_type, f"Observation: You see a {instance_type.split('.')[0]}."
            )
            action = {
                "action": f"look_at('{random_instance}')",
                "result": result_text,
            }
        else:
            if not generic_action_pool:
                break
            action_name = random.choice(list(generic_action_pool.keys()))
            action = _create_action(action_name, all_object_instances)
        random_actions.append(action)

    # 4. Combine, shuffle, and format
    action_sequence = specific_actions + random_actions
    random.shuffle(action_sequence)

    formatted_output = []
    for i, item in enumerate(action_sequence):
        formatted_output.append(f"Take Action {i+1}:")
        formatted_output.append(f"  Action: {item['action']}")
        formatted_output.append(f"  Return: {item['result']}")

    return "\n".join(formatted_output)


def _select_objects(
    objects_config,
    is_furniture,
    max_count,
    must_include_types,
    goal_type=None,
):
    """Helper to select furniture or small items based on constraints."""
    # Create a pool of all potential items of the given category
    potential_pool = [
        obj_type
        for obj_type, props in objects_config.items()
        if props.get("is_furniture", False) == is_furniture
        for _ in range(props.get("count", 0))
    ]

    # Start with must-include types, ensuring no duplicates
    selected_raw = list(set(must_include_types))

    # Add the goal type if it's not already included
    if goal_type and goal_type not in selected_raw:
        selected_raw.append(goal_type)

    # Fill remaining slots from the general pool
    remaining_slots = max_count - len(selected_raw)
    if remaining_slots > 0:
        # Exclude already selected items from the general pool
        general_pool = [item for item in potential_pool if item not in selected_raw]
        random.shuffle(general_pool)
        selected_raw.extend(general_pool[:remaining_slots])

    # Count the final selection
    final_counts = defaultdict(int)
    for item_type in selected_raw:
        final_counts[item_type] += 1

    return final_counts


def _place_item(
    item_inst,
    obj_type,
    properties,
    available_furniture,
    furniture_props,
    must_include_containers,
):
    """Determines the placement for a single small item."""
    # Prioritize must-include containers if the item is a must-include item
    if obj_type in must_include_containers:
        for cont_type in must_include_containers:
            if cont_type in available_furniture:
                loc_inst = random.choice(available_furniture[cont_type])
                is_container = furniture_props.get(cont_type, {}).get(
                    "is_container", False
                )
                if is_container and properties.get("inside_sources"):
                    return loc_inst, "inside"
                if not is_container and properties.get("ontop_sources"):
                    return loc_inst, "ontop"

    # Try preferred ontop locations
    ontop_locs = [
        loc
        for t in properties.get("ontop_sources", [])
        for loc in available_furniture.get(t, [])
    ]
    if ontop_locs:
        return random.choice(ontop_locs), "ontop"

    # Try preferred inside locations
    inside_locs = [
        loc
        for t in properties.get("inside_sources", [])
        for loc in available_furniture.get(t, [])
    ]
    if inside_locs:
        return random.choice(inside_locs), "inside"

    # Fallback to any available furniture
    all_furniture = [
        inst for inst_list in available_furniture.values() for inst in inst_list
    ]
    if all_furniture:
        loc_inst = random.choice(all_furniture)
        loc_type = loc_inst.rsplit("_", 1)[0]
        is_container = furniture_props.get(loc_type, {}).get("is_container", False)
        return loc_inst, "inside" if is_container else "ontop"

    # Final fallback to the floor
    return "floor.n.01_1", "onfloor"


def _adjust_placements(
    object_placements,
    init_predicates,
    available_furniture,
    furniture_props,
    min_items_per_container,
    goal_item_inst,
    goal_container_inst,
    num_other_items_in_sensitive_container: int = 3,
):
    """Adjusts item placements to meet clustering and goal requirements."""
    all_furniture_insts = [
        inst for inst_list in available_furniture.values() for inst in inst_list
    ]

    # 1. Ensure minimum items per container
    if min_items_per_container > 0:
        for container_inst in all_furniture_insts:
            items_on_container = [
                obj for obj, cont in object_placements.items() if cont == container_inst
            ]
            needed = min_items_per_container - len(items_on_container)
            if needed > 0:
                movable = [
                    obj
                    for obj in object_placements
                    if object_placements[obj] != container_inst
                ]
                random.shuffle(movable)
                for obj_to_move in movable[:needed]:
                    _move_item(
                        obj_to_move,
                        container_inst,
                        object_placements,
                        init_predicates,
                        furniture_props,
                    )

    # 2. Ensure goal item and its companions are on the goal container
    if goal_item_inst and goal_container_inst:
        # Move the goal item itself if it's not in the right place
        if object_placements.get(goal_item_inst) != goal_container_inst:
            _move_item(
                goal_item_inst,
                goal_container_inst,
                object_placements,
                init_predicates,
                furniture_props,
            )

        # Move other items to the goal container to have at least 3 items there
        items_on_goal_cont = [
            obj
            for obj, cont in object_placements.items()
            if cont == goal_container_inst
        ]
        needed = num_other_items_in_sensitive_container - len(items_on_goal_cont)
        if needed > 0:
            movable = [
                obj
                for obj in object_placements
                if object_placements[obj] != goal_container_inst
                and obj != goal_item_inst
            ]
            random.shuffle(movable)
            for obj_to_move in movable[:needed]:
                _move_item(
                    obj_to_move,
                    goal_container_inst,
                    object_placements,
                    init_predicates,
                    furniture_props,
                )


def _move_item(
    item_inst, new_container_inst, object_placements, init_predicates, furniture_props
):
    """Helper to move an item from its old container to a new one."""
    old_container = object_placements.get(item_inst)
    if old_container:
        # Remove old placement predicate
        for pred in list(init_predicates):
            if pred.endswith(f"{item_inst} {old_container})"):
                init_predicates.remove(pred)
                break
    # Add new placement predicate
    new_container_type = new_container_inst.rsplit("_", 1)[0]
    is_container = furniture_props.get(new_container_type, {}).get(
        "is_container", False
    )
    pred_type = "inside" if is_container else "ontop"
    init_predicates.add(f"({pred_type} {item_inst} {new_container_inst})")
    object_placements[item_inst] = new_container_inst


def generate_home_room_pddl_scenario(
    room_name="living_room",
    objects_config: dict = None,
    goal_item_type="note.n.01",
    goal_container_type="table.n.02",
    total_small_items=20,
    max_furniture_amount=5,
    min_small_items_per_container=1,
    must_include_item_types: list = None,
    must_include_container_types: list = None,
    custom_initial_state: list = None,
    custom_goal_str: str = None,
    num_other_items_in_sensitive_container: int = 3,
):
    """
    Generates a PDDL problem file for a home room scenario.
    """
    if objects_config is None:
        objects_config = DEFAULT_OBJECT_CONFIG
    if must_include_item_types is None:
        must_include_item_types = []
    if must_include_container_types is None:
        must_include_container_types = []

    # --- 1. Initialize Scene ---
    problem_name = f"home_organization_scenario_{random.randint(1000, 9999)}"
    domain_name = "igibson"
    all_instances_by_type = {
        "floor.n.01": ["floor.n.01_1"],
        "agent.n.01": ["agent.n.01_1"],
    }
    init_predicates = {
        f"(onfloor agent.n.01_1 floor.n.01_1)",
        f"(inroom agent.n.01_1 {room_name})",
        f"(inroom floor.n.01_1 {room_name})",
    }

    if custom_initial_state:
        init_predicates.update(custom_initial_state)

    # --- 2. Select and Instantiate Furniture ---
    final_furniture_counts = _select_objects(
        objects_config,
        is_furniture=True,
        max_count=max_furniture_amount,
        must_include_types=must_include_container_types,
        goal_type=goal_container_type,
    )

    available_furniture = defaultdict(list)
    furniture_props = {}
    for obj_type, count in final_furniture_counts.items():
        instances = [f"{obj_type}_{i+1}" for i in range(count)]
        all_instances_by_type[obj_type] = instances
        available_furniture[obj_type].extend(instances)
        furniture_props[obj_type] = {
            "is_container": objects_config[obj_type].get("is_container", False)
        }
        for inst in instances:
            init_predicates.add(f"(onfloor {inst} floor.n.01_1)")
            init_predicates.add(f"(inroom {inst} {room_name})")

    # --- 3. Select and Instantiate Small Items ---
    final_small_item_counts = _select_objects(
        objects_config,
        is_furniture=False,
        max_count=total_small_items,
        must_include_types=must_include_item_types,
        goal_type=goal_item_type,
    )

    object_placements = {}
    for obj_type, count in final_small_item_counts.items():
        instances = [f"{obj_type}_{i+1}" for i in range(count)]
        all_instances_by_type[obj_type] = instances
        for item_inst in instances:
            init_predicates.add(f"(inroom {item_inst} {room_name})")
            loc_inst, pred_type = _place_item(
                item_inst,
                obj_type,
                objects_config[obj_type],
                available_furniture,
                furniture_props,
                must_include_container_types,
            )
            init_predicates.add(f"({pred_type} {item_inst} {loc_inst})")
            object_placements[item_inst] = loc_inst

    # --- 4. Adjust Placements for Clustering and Goal ---
    goal_item_inst = all_instances_by_type.get(goal_item_type, [None])[0]
    goal_container_inst = all_instances_by_type.get(goal_container_type, [None])[0]

    _adjust_placements(
        object_placements,
        init_predicates,
        available_furniture,
        furniture_props,
        min_small_items_per_container,
        goal_item_inst,
        goal_container_inst,
        num_other_items_in_sensitive_container,
    )

    # --- 5. Define Goal ---
    goal_predicates = []
    if custom_goal_str:
        goal_predicates.append(custom_goal_str)
    elif goal_item_inst and goal_container_inst:
        is_container = furniture_props.get(goal_container_type, {}).get(
            "is_container", False
        )
        pred_type = "inside" if is_container else "ontop"
        goal_predicates.append(f"({pred_type} {goal_item_inst} {goal_container_inst})")
    else:
        goal_predicates.append(f"(inroom agent.n.01_1 {room_name})")

    # --- 6. Assemble PDDL ---
    pddl_lines = [
        f"(define (problem {problem_name})",
        f"(:domain {domain_name})",
        "(:objects",
    ]
    for obj_type, instances in sorted(all_instances_by_type.items()):
        pddl_lines.append(f"  {' '.join(instances)} - {obj_type}")
    pddl_lines.extend(
        [
            ")",
            "(:init",
            *[f"  {pred}" for pred in sorted(list(init_predicates))],
            ")",
            "(:goal",
        ]
    )
    if len(goal_predicates) > 1 or custom_goal_str:
        pddl_lines.append("  (and")
        pddl_lines.extend([f"    {pred}" for pred in sorted(goal_predicates)])
        pddl_lines.append("  )")
    else:
        pddl_lines.append(f"  {goal_predicates[0]}")
    pddl_lines.extend([")", ")"])

    return "\n".join(pddl_lines), all_instances_by_type, object_placements
