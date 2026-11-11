# make_workloads.py  — run once, output the three files

import json, random



def write(path, rows):

    with open(path, "w") as f:

        for r in rows:

            f.write(json.dumps(r) + "\n")



# light: 50 short sessions, 3-6 turns, small tool outputs

light = [

    {

        "num_turns": random.randint(3, 6),

        "input_prompt_length": random.randint(48, 96),

        "assistant_response_length": [32] * 6,

        "tool_call_output_length": [random.randint(32, 96)] * 6,

        "tool_call_latency": [0.2] * 6,

        "final_assistant_response_length": 64,

    }

    for _ in range(50)

]



# heavy: 50 long sessions, 8-12 turns, large tool outputs (growing context)

heavy = [

    {

        "num_turns": 6,

        "input_prompt_length": 128,

        "assistant_response_length": [64] * 6,

        "tool_call_output_length": [384] * 6,

        "tool_call_latency": [0.0] * 6,

        "final_assistant_response_length": 128,

    }

    for _ in range(50)

]



# slow-tools: 50 sessions with 2s tool latency (tests scheduler idle gaps)

slow = [

    {

        "num_turns": 6,

        "input_prompt_length": 64,

        "assistant_response_length": [48] * 6,

        "tool_call_output_length": [256] * 6,

        "tool_call_latency": [2.0] * 6,

        "final_assistant_response_length": 64,

    }

    for _ in range(50)

]



write("workloads/light.jsonl", light)

write("workloads/heavy.jsonl", heavy)

write("workloads/slow-tools.jsonl", slow)
