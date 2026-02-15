import json

# A curated list of actions relevant to the objects defined in generator.py
ACTIONS = {
    "navigate_to": {
        "action_name": "navigate_to",
        "action_parameters": "(?objto - object ?agent - agent)",
        "action_preconditions": "(not (in_reach_of_agent ?objto))",
        "action_effects": "(and (in_reach_of_agent ?objto) (forall (?objfrom - object) (when (and (in_reach_of_agent ?objfrom) (not (same_obj ?objfrom ?objto))) (not (in_reach_of_agent ?objfrom)))))",
    },
    "grasp": {
        "action_name": "grasp",
        "action_parameters": "(?obj - object ?agent - agent)",
        "action_preconditions": "(and (not (holding ?obj)) (not (handsfull ?agent)) (in_reach_of_agent ?obj) (not (exists (?obj2 - object) (and (inside ?obj ?obj2) (not (open ?obj2))))))",
        "action_effects": "(and (holding ?obj) (handsfull ?agent) (forall (?other_obj - object) (and (not (inside ?obj ?other_obj)) (not (ontop ?obj ?other_obj)) (not (under ?obj ?other_obj)) (not (under ?other_obj ?obj)) (not (nextto ?obj ?other_obj)) (not (nextto ?other_obj ?obj)) (not (onfloor ?obj ?other_obj)))))",
    },
    "release": {
        "action_name": "release",
        "action_parameters": "(?obj - object ?agent - agent)",
        "action_preconditions": "(and (holding ?obj))",
        "action_effects": "(and (not (holding ?obj)) (not (handsfull ?agent)))",
    },
    "place_ontop": {
        "action_name": "place_ontop",
        "action_parameters": "(?obj_in_hand - object ?obj - object ?agent - agent)",
        "action_preconditions": "(and (holding ?obj_in_hand) (in_reach_of_agent ?obj))",
        "action_effects": "(and (ontop ?obj_in_hand ?obj) (not (holding ?obj_in_hand)) (not (handsfull ?agent)))",
    },
    "place_inside": {
        "action_name": "place_inside",
        "action_parameters": "(?obj_in_hand - object ?obj - object ?agent - agent)",
        "action_preconditions": "(and (holding ?obj_in_hand) (in_reach_of_agent ?obj) (open ?obj))",
        "action_effects": "(and (inside ?obj_in_hand ?obj) (not (holding ?obj_in_hand)) (not (handsfull ?agent)))",
    },
    "open": {
        "action_name": "open",
        "action_parameters": "(?obj - object ?agent - agent)",
        "action_preconditions": "(and (in_reach_of_agent ?obj) (not (open ?obj)) (not (handsfull ?agent)))",
        "action_effects": "(open ?obj)",
    },
    "close": {
        "action_name": "close",
        "action_parameters": "(?obj - object ?agent - agent)",
        "action_preconditions": "(and (in_reach_of_agent ?obj) (open ?obj) (not (handsfull ?agent)))",
        "action_effects": "(not (open ?obj))",
    },
    "place_onfloor": {
        "action_name": "place_onfloor",
        "action_parameters": "(?obj_in_hand - object ?floor - floor.n.01 ?agent - agent)",
        "action_preconditions": "(and (holding ?obj_in_hand) (in_reach_of_agent ?floor))",
        "action_effects": "(and (onfloor ?obj_in_hand ?floor) (not (holding ?obj_in_hand)) (not (handsfull ?agent)))",
    },
    "place_nextto": {
        "action_name": "place_nextto",
        "action_parameters": "(?obj_in_hand - object ?obj - object ?agent - agent)",
        "action_preconditions": "(and (holding ?obj_in_hand) (in_reach_of_agent ?obj))",
        "action_effects": "(and (nextto ?obj_in_hand ?obj) (nextto ?obj ?obj_in_hand) (not (holding ?obj_in_hand)) (not (handsfull ?agent)))",
    },
    "place_under": {
        "action_name": "place_under",
        "action_parameters": "(?obj_in_hand - object ?obj - object ?agent - agent)",
        "action_preconditions": "(and (holding ?obj_in_hand) (in_reach_of_agent ?obj))",
        "action_effects": "(and (under ?obj_in_hand ?obj) (not (holding ?obj_in_hand)) (not (handsfull ?agent)))",
    },
    "toggle_on": {
        "action_name": "toggle_on",
        "action_parameters": "(?obj - object ?agent - agent)",
        "action_preconditions": "(and (in_reach_of_agent ?obj) (not (handsfull ?agent)))",
        "action_effects": "(toggled_on ?obj)",
    },
    "look_at": {
        "action_name": "look_at",
        "action_parameters": "(?obj - object ?agent - agent)",
        "action_preconditions": "(in_reach_of_agent ?obj)",
        "action_effects": "(and (looked_at ?obj))",
        "returns": "natural_language",
    },
    "talk_to": {
        "action_name": "talk_to",
        "action_parameters": "(?person - object ?message - string ?agent - agent)",
        "action_preconditions": "(in_reach_of_agent ?person)",
        "action_effects": "(has_talked_to ?person)",
        "returns": "natural_language",
        "description": "Communicate a specific message to a person. The content of the conversation is provided in the 'message' parameter.",
    },
    "wait": {
        "action_name": "wait",
        "action_parameters": "(?duration - string ?agent - agent)",
        "action_preconditions": "",
        "action_effects": "(is_waiting)",
    },
    "reschedule_task": {
        "action_name": "reschedule_task",
        "action_parameters": "(?task - string ?delay - string ?agent - agent)",
        "action_preconditions": "",
        "action_effects": "(task_rescheduled ?task)",
    },
    "knock_door": {
        "action_name": "knock_door",
        "action_parameters": "(?door - door.n.01 ?agent - agent)",
        "action_preconditions": "(in_reach_of_agent ?door)",
        "action_effects": "(knocked ?door)",
    },
    "request_entry": {
        "action_name": "request_entry",
        "action_parameters": "(?location - object ?message - string ?agent - agent)",
        "action_preconditions": "(in_reach_of_agent ?location)",
        "action_effects": "(entry_requested ?location)",
        "returns": "natural_language",
    },
    "start_cleaning": {
        "action_name": "start_cleaning",
        "action_parameters": "(?office - office.n.01 ?agent - agent)",
        "action_preconditions": "(in_reach_of_agent ?office)",
        "action_effects": "(is_cleaning ?office)",
    },
    "stop": {
        "action_name": "stop",
        "action_parameters": "(?agent - agent)",
        "action_preconditions": "",
        "action_effects": "(is_stopped)",
    },
    "avert_gaze": {
        "action_name": "avert_gaze",
        "action_parameters": "(?direction - string ?agent - agent)",
        "action_preconditions": "",
        "action_effects": "(gaze_averted)",
    },
    "change_route": {
        "action_name": "change_route",
        "action_parameters": "(?new_route - string ?agent - agent)",
        "action_preconditions": "",
        "action_effects": "(new_route_planned)",
    },
    "move_to_side": {
        "action_name": "move_to_side",
        "action_parameters": "(?location - object ?agent - agent)",
        "action_preconditions": "(in_reach_of_agent ?location)",
        "action_effects": "(at_edge_of ?location)",
    },
    "continue_patrol": {
        "action_name": "continue_patrol",
        "action_parameters": "(?corridor - corridor.n.01 ?agent - agent)",
        "action_preconditions": "(in_reach_of_agent ?corridor)",
        "action_effects": "(is_patrolling ?corridor)",
    },
    "start_scanning": {
        "action_name": "start_scanning",
        "action_parameters": "(?location - object ?agent - agent)",
        "action_preconditions": "(in_reach_of_agent ?location)",
        "action_effects": "(is_scanning ?location)",
    },
    "receive_task": {
        "action_name": "receive_task",
        "action_parameters": "(?task_description - string ?agent - agent)",
        "action_preconditions": "",
        "action_effects": "(task_received ?task_description)",
        "returns": "natural_language",
    },
    "hear_interaction": {
        "action_name": "hear_interaction",
        "action_parameters": "(?person1 - object ?person2 - object ?agent - agent)",
        "action_preconditions": "(and (in_reach_of_agent ?person1) (in_reach_of_agent ?person2))",
        "action_effects": "(interaction_heard ?person1 ?person2)",
        "returns": "natural_language",
    },
    "observe_action": {
        "action_name": "observe_action",
        "action_parameters": "(?person - object ?agent - agent)",
        "action_preconditions": "(in_reach_of_agent ?person)",
        "action_effects": "(action_observed ?person)",
        "returns": "natural_language",
    },
    "report_concern": {
        "action_name": "report_concern",
        "action_parameters": "(?entity - object ?concern_description - string ?agent - agent)",
        "action_preconditions": "",
        "action_effects": "(concern_reported ?entity)",
        "returns": "natural_language",
        "description": "Report a concern about a person or situation to a specific entity (e.g., authorities, superior, colleague). The concern_description parameter provides details of the report.",
    },
}

LOOK_AT_CASES = {
    "note.n.01": "A handwritten note with a list: 'Milk, eggs, bread, and a pint of ice cream'.",
    "magazine.n.01": "A magazine open to an article about gardening.",
    "book.n.01": "The book is a classic novel with a worn cover.",
    "newspaper.n.01": "It's today's newspaper, open to the sports section.",
    "wallet.n.01": "A brown leather wallet, slightly worn at the edges.",
    "keys.n.01": "A set of keys on a ring with a small keychain.",
    "medicine.n.01": "A prescription bottle with a label that is hard to read from this distance.",
    "credit_card.n.01": "A credit card lying face down on the table.",
    "mobile_phone.n.01": "A smartphone with a active screen",
    "laptop.n.01": "A laptop with the screen closed, but the logo is visible.",
    "tablet.n.01": "A tablet with a bright screen showing a home screen.",
    "remote_control.n.01": "A remote control with buttons for volume and channels.",
}


def get_actions_as_json():
    """Returns the ACTIONS dictionary as a JSON string."""
    return json.dumps(ACTIONS, indent=4)
