from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1",api_key="dummy",)
# --- without thinking (fast) ---
resp = client.chat.completions.create(model="Qwen/Qwen3.5-27B",messages=[{"role": "user", "content": "Write a sentence about LLMs"}],temperature=0.0,max_tokens=64)
print("NO THINKING:")
print(resp.choices[0].message.content)
# from openai import OpenAI
# client = OpenAI(base_url="http://127.0.0.1:8001/v1",api_key="dummy",)
# # --- without thinking (fast) ---
# resp = client.chat.completions.create(model="google/gemma-4-31B-it",messages=[{"role": "user", "content": "Write a sentence about LLMs"}],temperature=0.0,max_tokens=64)
# print("NO THINKING:")
# print(resp.choices[0].message.content)

import argparse
import csv
import os
import sys
import re
import time
import random

import pandas as pd

from openai import OpenAI


import spot
#spot.setup()


import json

timestamp = int(time.time())
from pathlib import Path
current_tempOut = Path("tempOut/"+str(timestamp))
print(timestamp)
#tempOut = Path("tempOut/")
#tempOut.mkdir()
current_tempOut.mkdir()
current_tempOut = str(current_tempOut)+"/"
INSTRUCTION = """
Return EXACTLY one LTL formula as a single line.
The output must be directly parseable as an LTL formula using the Spot API.
"""

BASIC = """
You are a Linear Temporal Logic ( LTL ) Parser. Your task is to convert a given natural language statement to an LTL formula, using the provided mapping of natural language phrases to atomic propositions.

LTL Symbols :
    - AND : &
    - OR : |
    - NOT : !
    - IMPLIES : ->
    - BIIMPLICATION : <->
    - NEXT : X
    - EVENTUALLY : F
    - ALWAYS : G
    - UNTIL : U

Natural Language statement : {requirement}
Atomic Propositions : {atomic_proposition}
"""

# Direct TL variant"
ARTEMIS = """
You are an expert in translating natural language to linear temporal logic (LTL). Your job is to translate natural language to LTL. You must only use LTL operators and atomic propositions (NO NUMERICAL COMPARISON OPERATORS ALLOWED).
Recall that in LTL, G = globally, F = eventually, V = releases, X = next, U = until. You may use boolean operators (e.g., !, &, |, ->, <->) and can only use atomic propositions (NO NUMERICAL COMPARISON OPERATORS ALLOWED).

Inputs consist of:
1. unstructured natural language (string)
2. atomic proposition dictionary
    
The Outputs consist of:
1. output_LTL
    
Provide a list of the top 1 most likely translations (ordered by most likely first to least likely last) in the above output format for the following:

    'input_natural_language':{requirement},
    'atomic_propositions':{atomic_proposition}

"""

ADARULE = """
Task:
Translate natural language (NL) sentences into Linear Temporal Logic (LTL) formulas accurately.
Your answers always need to follow the output format.

Rules:
The converted formula should only contain atomic statements and operators.
Use standard LTL syntax and operators: G (globally), F (eventually), X (next), U (until), R (release), ! (negation), & (conjunction), | (disjunction), -> (implication), <-> (equivalence).
G means "globally": G a indicates that a is true in all future states.
F means "finally": F a indicates that a will eventually be true in some future state.
X means "next": X a is true if a is true in the next state.
U means "until": a U b is true if a remains true until b becomes true.
R means "release": a R b means b must be true until the moment when a is true and b is true. Once a is true, b can no longer be true. If a is never true, then b must always remain true.
Remember especially that the brackets match, we stipulate that each atomic formula is followed by a space. 
Do not change atomic propositions in NL.


Guidelines:
When translate "never", use  G!;
When translate "every time", "always", "all the time", use  G;
When translate "at certain moment", "eventually", "in the futhure" , "sooner or later", use  F;
When expressing "both A and B" or A and B will happen together at some moment, use A ∧ B;
When translate  "A or B holds"  or at some moment at least one of  A , B will be true , use A | B;
When the sentence is a discription of the system state, it means the state is always satisfied, so use G;
When the sentence is a discription of a and b happens, then finally,c and d will happen, you should considered the situation that the state "a and b" never satisfied , (e.g. G(!(a & b)) );
When translate "it is going to happen that a after b", use b -> F (a);
When translate  "Never (a) after (b)",use G((b) -> G(!(a)));
When translate  "Whenever (a), then (b)", use G((b) -> (a));
When translate  "A never happens", use G(!A);
For "After A, B happens"， Weak interpretation (B may happen even if A doesn’t)， use G(A -> F(B)),  Strong interpretation (A must happen first, then B), use !A | F(A & F(B));
For sentences like "Whenever A or B, then eventually C or D", use G((A | B) -> F(C | D));
For sentences like "At some point (A), and later (B)", use F(A & F(B));
For "First A and B, then eventually C or D" , use !(A & B) | F((A & B) & F(C | D));
For "After A and B, eventually C", use G((A & B) -> F(C));

Natural Language: {requirement}
Atomic Propositions: {atomic_proposition}

Please response in plain text format. DO NOT use markdown, latex or any other formats.
Please response in the following format, and replace the '[LTL formula]' with the LTL formula translated from the natural language sentences:
So the final LTL translation is: [LTL formula].FINISH
"""

def normalize_formula(text: str) -> str:
    """Strip common formatting so the result is parser-friendly."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Keep only the first non-empty line if the model accidentally adds extra text.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def ask_chatgpt_gemma(client: OpenAI, model: str, prompt: str, requirement: str, atomic_proposition: str) -> str:

    # --- without thinking (fast) ---
    response = client.chat.completions.create(
        model=model,
        temperature=temp,
        messages=[
            {
                "role": "user",
                "content": globals()[prompt].format(
                    requirement=requirement,
                    atomic_proposition=atomic_proposition,
                ) + INSTRUCTION,
            }
        ],
    )

    return normalize_formula(response.choices[0].message.content or "")


def ask_chatgpt(client: OpenAI, model: str, prompt: str, requirement: str, atomic_proposition: str) -> str:

    # --- without thinking (fast) ---
    content = globals()[prompt].format(
                    requirement=requirement,
                    atomic_proposition=atomic_proposition,
                ) + INSTRUCTION
    
    response = client.chat.completions.create(
        model=model,
        temperature=temp,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        extra_body={
        "chat_template_kwargs": {
        "enable_thinking": False
        }
        },
    )

    return normalize_formula(response.choices[0].message.content or "")


def semantically_equivalent(formula_a: str, formula_b: str):
    """
    Returns:
        True  -> semantically equivalent
        False -> valid syntax but not equivalent
        None  -> syntax error, exclude from accuracy
    """
    try:

        f_a = spot.formula(formula_a)
        f_b = spot.formula(formula_b)

        xor_formula = spot.formula.Not(spot.formula.Equiv(f_a, f_b))
        return spot.translate(xor_formula).is_empty()

    except Exception as exc:
        msg = str(exc)

        if "syntax error" in msg.lower():
            with open(current_tempOut+"output_log.txt", "a", encoding="utf-8") as f:
                print(
                    f"Syntax error; excluding from accuracy:\n"
                    f"  Error:        {exc}",
                    file=f,
                )
            print(
                f"Syntax error; excluding from accuracy:\n"
                f"  Error:        {exc}",
                file=sys.stderr,
            )
            return None
        with open(current_tempOut+"output_log.txt", "a", encoding="utf-8") as f:
            print(
                f"Warning: could not compare formulas:\n"
                f"  Error:        {exc}",
                file=f,
            )
        print(
        f"Warning: could not compare formulas:\n"
        f"  Error:        {exc}",
        file=sys.stderr,
        )
        return None



prompts = ["BASIC", "ARTEMIS" , "ADARULE"]



file_input="HumanDataProcessed/"
model = "Qwen/Qwen3.5-27B"


import time

for file_inp in os.listdir(file_input):

    df = pd.read_csv(file_input+file_inp, sep=',')
    df['Index'] = df.index  

    for temp in [ 0.7 ]: #
        print(f"Temperature: {temp}")


        for prompt in prompts:
            dataset = []    
            output = []
            model_filename = model.replace("/","_")
            index_set = set()

            output_file = "out_vagueness_human/"+f"data_{file_inp}_{model_filename}_{prompt}_temp{int(temp*100)}.csv"
            full_output_file = "full_outputs/" + f"data_{file_inp}_{model_filename}_{prompt}_temp{int(temp*100)}.csv"
            if not os.path.exists(output_file):
                with open(output_file, "w", newline="", encoding="utf-8") as g:
                    writer = csv.DictWriter(
                        g,
                        fieldnames=['PROMPTTYPE',"Index", "Requirement", "Ground Truth", "Response", "Equivalent"],
                    )
                    writer.writeheader()
                


            already_processed = set()
            if os.path.exists(full_output_file):
                my_headers = ['PROMPTTYPE',"Index", "Requirement", "Ground Truth", "Atomic Proposition", "Original Response", "Response", "Equivalent"]
                with open(full_output_file, "r", encoding="utf-8") as f:
                    
                    reader = csv.DictReader(f, fieldnames=my_headers)
                    for row in reader:
                        index= row['Index']
                        already_processed.add(int(index))

            for ind, row in df.iterrows():
                requirement = str(row.iloc[0])
                ground_truth = str(row.iloc[1]).strip()
                atomic_proposition = str(row.iloc[2]).strip()

                if int(ind) not in already_processed:
                    dataset.append((ind,requirement, ground_truth, atomic_proposition))

            print("start",len(dataset), dataset[0])
            for ind, requirement, ground_truth, atomic_proposition in dataset:
                parse_errors = 0
                syntax_errors = 0
                total = 0
                correct = 0
                rows = []
                print(ind, requirement, ground_truth, atomic_proposition)
                for retake in range(100):
                    print(retake)
                    client = OpenAI(base_url="http://127.0.0.1:8000/v1",api_key="dummy",)
                    
                    if prompt == "BASIC":
                        model_response = ask_chatgpt(client, model, prompt, requirement, atomic_proposition)
                        og_model_response = model_response
                    if prompt == "ADARULE":
                        model_response = ask_chatgpt(client, model, prompt, requirement, atomic_proposition)
                        og_model_response = model_response
                        
                        model_response = model_response.replace("So the final LTL translation is: ", "")
                        model_response = model_response.replace(".FINISH", "").strip()

                    if prompt == "ARTEMIS":    
                        model_response = ask_chatgpt(client, model, prompt, requirement, atomic_proposition)
                        og_model_response = model_response

                    equivalent = semantically_equivalent(ground_truth, model_response)

                    equivalent2 = equivalent
                    if equivalent is None:
                        equivalent2 = "None"

                    output.append(
                        {       
                                                    
                                'PROMPTTYPE': prompt,
                                "Index": ind,
                                "Requirement": requirement,
                                "Ground Truth": ground_truth,
                                "Atomic Proposition": atomic_proposition,
                                "Original Response": og_model_response,
                                "Response": model_response,
                                "Equivalent": equivalent2
                            }
                    )

                    
                    if equivalent is None:
                        syntax_errors += 1
                    else:
                        total += 1
                        correct += int(equivalent)

                    rows.append(
                            {   
                                'PROMPTTYPE': prompt,
                                "Index": ind,
                                "Requirement": requirement,
                                "Ground Truth": ground_truth,
                                "Response": model_response,
                                "Equivalent": equivalent2
                            }
                        )
                            






                

                accuracy = correct / total if total else 0.0
                print(f"{ind}",accuracy, correct ,total)

                unique_rows = []
                unique_ltl = []
                for x in rows:
                    try:
                        y = str(spot.formula(x["Response"]))
                    except Exception as e:
                        y = x["Response"]

                    if y not in unique_ltl:
                        unique_ltl.append(y)
                        unique_rows.append(x)
                
                if len(unique_rows) > 1:
                    with open(current_tempOut+"loggs_accuracy.csv", "a", newline="", encoding="utf-8") as g:
                        print(f"Total accuracy: {accuracy:.4f} ({correct}/{total})",file=g,)
                        print(f"Syntax errors excluded: {syntax_errors}",file=g,)
                    print(f"Total accuracy: {accuracy:.4f} ({correct}/{total})")
                    print(f"Syntax errors excluded: {syntax_errors}")
                    with open(output_file, "a", newline="", encoding="utf-8") as g:
                        writer = csv.DictWriter(
                            g,
                            fieldnames=['PROMPTTYPE',"Index", "Requirement", "Ground Truth", "Response", "Equivalent"],
                        )
                        writer.writerows(rows)
                
                with open(full_output_file, "w", newline="", encoding="utf-8") as g:
                        writer = csv.DictWriter(
                            g,
                            fieldnames=['PROMPTTYPE',"Index", "Requirement", "Ground Truth", "Atomic Proposition", "Original Response", "Response", "Equivalent"],
                        )
                        writer.writerows(output)
