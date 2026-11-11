#!/usr/bin/env python3

import json, sys, argparse



REQUIRED = [

    "input_prompt_length", "num_turns", "assistant_response_length",

    "tool_call_output_length", "tool_call_latency", "final_assistant_response_length",

]



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("path")

    ap.add_argument("--max-model-len", type=int, default=4096)

    a = ap.parse_args()

    bad = 0

    for i, line in enumerate(open(a.path), 1):

        line = line.strip()

        if not line:

            continue

        try:

            r = json.loads(line)

        except Exception as e:

            print(f"{a.path}:{i}: JSON error {e}")

            bad += 1

            continue



        missing = [k for k in REQUIRED if k not in r]

        if missing:

            print(f"{a.path}:{i}: missing {missing}")

            bad += 1

            continue



        n = r["num_turns"]

        for k in ["assistant_response_length", "tool_call_output_length", "tool_call_latency"]:

            if not isinstance(r[k], list) or len(r[k]) != n:

                print(f"{a.path}:{i}: {k} len {len(r.get(k, []))} != num_turns {n}")

                bad += 1



        if not isinstance(r["input_prompt_length"], int):

            print(f"{a.path}:{i}: input_prompt_length must be int")

            bad += 1

        if not isinstance(r["final_assistant_response_length"], int):

            print(f"{a.path}:{i}: final_assistant_response_length must be int")

            bad += 1



        ctx = (r["input_prompt_length"]

               + sum(r["assistant_response_length"])

               + sum(r["tool_call_output_length"]))

        if ctx > a.max_model_len:

            print(f"{a.path}:{i}: WARN cumulative context {ctx} > max-model-len {a.max_model_len}")



    if bad:

        print(f"FAIL {bad} bad row(s) in {a.path}")

        sys.exit(1)

    print(f"OK {a.path}")



if __name__ == "__main__":

    main()
