import sqlite3
import time
import json
import random
import itertools
from multiprocessing import Process, cpu_count
import argparse
import csv
import os
import sys
import re
from pathlib import Path
import pandas as pd
from openai import OpenAI
import spot

# Initialize directories
timestamp = int(time.time())
current_tempOut = Path("tempOut/" + str(timestamp))
current_tempOut.mkdir(parents=True, exist_ok=True)
current_tempOut_str = str(current_tempOut) + "/"

# Global Prompts
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

model = "google/gemma-4-31B-it"
# model = "Qwen/Qwen3.5-27B"
DB_PATH = "human_data_gemma_07.db"


STALL_TIMEOUT_SECONDS = 600  

def normalize_formula(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""

def ask_chatgpt_QWEN(client: OpenAI, model: str, prompt: str, requirement: str, atomic_proposition: str, temperature: float) -> str:
    content = globals()[prompt].format(
        requirement=requirement,
        atomic_proposition=atomic_proposition,
    ) + INSTRUCTION
    
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return normalize_formula(response.choices[0].message.content or "")

def ask_chatgpt_gemma(client: OpenAI, model: str, prompt: str, requirement: str, atomic_proposition: str, temperature: float) -> str:

    # --- without thinking (fast) ---
    response = client.chat.completions.create(
        model=model,
        temperature=0,
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

def ask_chatgpt_API(client: OpenAI, prompt: str, requirement: str, atomic_proposition: str) -> str:

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        temperature=0,
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

def semantically_equivalent(formula_a: str, formula_b: str):
    try:
        f_a = spot.formula(formula_a)
        f_b = spot.formula(formula_b)
        xor_formula = spot.formula.Not(spot.formula.Equiv(f_a, f_b))
        return spot.translate(xor_formula).is_empty()
    except Exception as exc:
        msg = str(exc)
        if "syntax error" in msg.lower():
            with open(current_tempOut_str + "output_log.txt", "a", encoding="utf-8") as f:
                print(f"Syntax error; excluding from accuracy:\n  Error: {exc}", file=f)
            return None
        with open(current_tempOut_str + "output_log.txt", "a", encoding="utf-8") as f:
            print(f"Warning: could not compare formulas:\n  Error: {exc}", file=f)
        return None

# ==========================================
# 1. DATABASE SETUP & SEEDING
# ==========================================

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, isolation_level=None, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL') 
    return conn

def setup_database():
    with get_db_connection() as conn:
        conn.execute("BEGIN")
        conn.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_NL TEXT,
                original_LTL TEXT,
                Spot_LTL TEXT,
                APs TEXT,
                prompt_type TEXT,
                temperature REAL, -- Changed to REAL to handle decimals like 0.1
                status TEXT DEFAULT 'PENDING',
                started_at REAL,
                result_output TEXT,
                error_log TEXT,
                experiment_index INTEGER,
                UNIQUE(experiment_index, prompt_type, temperature)
            )
        ''')
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON experiments(status)")
        conn.execute("COMMIT")

def seed_experiments(data_points, prompt_types, temperatures):
    """Maps CSV dictionaries cleanly into SQL columns, checking for duplicates."""
    with get_db_connection() as conn:
        conn.execute("BEGIN")

        with get_db_connection() as conn:
            conn.execute("""
                DELETE FROM experiments 
                WHERE original_NL IS NULL 
                OR TRIM(original_NL) = '';
            """)

        print("Cleared rows with empty requirements.")
        
        for dp, pt, temp in itertools.product(data_points, prompt_types, temperatures):
            # Safe parsing from CSV keys
            exp_idx = int(dp['experiment_index'])
            exp_idx = int(dp['experiment_index'])
            orig_nl = dp.get('Requirement', '')
            orig_ltl = dp.get('Ground Truth', '')
            aps = dp.get('Atomic Proposition', '')
            try:
                spot_ltl = str(spot.formula(orig_ltl))
            except Exception as e:
                print(f"Error occurred while processing Spot_LTL for experiment {exp_idx}: {e}")
                spot_ltl = ''
            
            

            conn.execute('''
                INSERT OR IGNORE INTO experiments 
                (original_NL, original_LTL, Spot_LTL, APs, prompt_type, temperature, experiment_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (orig_nl, orig_ltl, spot_ltl, aps, pt, temp, exp_idx))
        try:  
            conn.execute("COMMIT")
        except Exception as e:
            print(f"Error during commit: {e}")

    print("✅ Database seeded with task combinations.")

def recover_stalled_tasks():
    stale_threshold = time.time() - STALL_TIMEOUT_SECONDS
    with get_db_connection() as conn:
        conn.execute("BEGIN")
        cursor = conn.execute('''
            UPDATE experiments 
            SET status = 'PENDING', started_at = NULL 
            WHERE status = 'PROCESSING' AND started_at < ?
        ''', (stale_threshold,))
        conn.execute("COMMIT")
        if cursor.rowcount > 0:
            print(f"⚠️ Recovered {cursor.rowcount} stalled tasks back to PENDING.")

# ==========================================
# 2. THE WORKER LOGIC
# ==========================================

def run_medium_duration_script(task):
    """Processes all 20 retakes in memory for a single parameter combination."""
    ind = task['experiment_index']
    requirement = task['original_NL']
    ground_truth = task['Spot_LTL']
    atomic_proposition = task['APs']
    prompt = task['prompt_type']
    temp = task['temperature']

    output = []
    client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="dummy")

    for retake in range(20):
        time.sleep(random.uniform(0.5, 1.5))  # Simulate processing time
        model_response = ask_chatgpt_QWEN(client, model, prompt, requirement, atomic_proposition, temp)
        og_model_response = model_response
        
        if prompt == "ADARULE":
            model_response = model_response.replace("So the final LTL translation is: ", "")
            model_response = model_response.replace(".FINISH", "").strip()

        equivalent = semantically_equivalent(ground_truth, model_response)
        equivalent2 = "None" if equivalent is None else equivalent

        output.append({       
            'PROMPTTYPE': prompt,
            "experiment_index": ind,
            "Requirement": requirement,
            "Ground Truth": ground_truth,
            "Atomic Proposition": atomic_proposition,
            "Original Response": og_model_response,
            "Response": model_response,
            "Equivalent": equivalent2
        })

    return output

def worker_process(worker_id):
    conn = get_db_connection()
    
    while True:
        try:
            # --- 1. ATOMIC TASK ACQUISITION ---
            conn.execute("BEGIN EXCLUSIVE")
            
            cursor = conn.execute('''
                SELECT * FROM experiments 
                WHERE status = 'PENDING' OR status = 'FAILED'
                LIMIT 1
            ''')
            task = cursor.fetchone()
            
            if not task:
                conn.execute("COMMIT")
                break 
                
            current_time = time.time()
            
            # Claim the task (Fixed tuple formatting and passed missing task ID)
            conn.execute('''
                UPDATE experiments 
                SET status = 'PROCESSING', started_at = ? 
                WHERE id = ?
            ''', (current_time, task['id']))
            
            conn.execute("COMMIT")
            
            # --- 2. EXECUTE THE WORK ---
            print(f"[Core {worker_id}] Started Task #{task['id']} (Index: {task['experiment_index']}, Prompt: {task['prompt_type']})")
            
            try:
                # Pass the complete database task row containing all required information
                result_data = run_medium_duration_script(task)
                
                # --- 3A. SAVE SUCCESS ---
                conn.execute("BEGIN")
                conn.execute('''
                    UPDATE experiments 
                    SET status = 'COMPLETED', result_output = ?, started_at = NULL 
                    WHERE id = ?
                ''', (json.dumps(result_data), task['id']))
                conn.execute("COMMIT")
                print(f"[Core {worker_id}] ✅ Completed Task #{task['id']}")
                
            except Exception as e:
                # --- 3B. SAVE FAILURE ---
                conn.execute("BEGIN")
                conn.execute('''
                    UPDATE experiments 
                    SET status = 'FAILED', error_log = ?, started_at = NULL 
                    WHERE id = ?
                ''', (str(e), task['id']))
                conn.execute("COMMIT")
                print(f"[Core {worker_id}] ❌ Failed Task #{task['id']}: {e}")

        except sqlite3.OperationalError:
            conn.execute("ROLLBACK")
            time.sleep(random.uniform(0.1, 0.5))

# ==========================================
# 3. THE MULTI-CORE ORCHESTRATOR
# ==========================================
import os
if __name__ == '__main__':
    setup_database()
    
    data_points = []
    # Make sure this path matches your directory setup
    file_input="HumanDataProcessed/"
    
    for fajl in os.listdir(file_input):
        with open(file_input+ fajl, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file, delimiter=',')
            for row in csv_reader:
                row['experiment_index'] = int(csv_reader.line_num)  # Ensure the index is an integer
                data_points.append(row)
            print(f"Loaded {csv_reader.line_num} rows from {fajl}.")
    print(f"Loaded {len(data_points)} data points from CSV files.")
    prompt_types = ["BASIC", "ARTEMIS", "ADARULE"]
    temperatures = [0.4]
    
    print("Initializing Database Seeding...")
    seed_experiments(data_points, prompt_types, temperatures)
    recover_stalled_tasks()
    
    num_cores = 10
    print(f"🚀 Spawning {num_cores} worker processes...")
    
    processes = []
    for i in range(num_cores):
        p = Process(target=worker_process, args=(i,))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("🎉 All processing complete! Outputs are safely stored in the database.")