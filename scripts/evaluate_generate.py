# evaluate_generate.py
import os
import torch
import pandas as pd
import numpy as np
import argparse
from functools import partial
from tqdm.auto import tqdm

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging,
)
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv() 
hf_home = os.getenv("HF_HOME")
if not hf_home:
    raise ValueError("FATAL: HF_HOME must be set in your .env file and loaded at the top of the script.")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
os.environ["HF_HUB_DISABLE_XET_STORAGE"] = "1"
print(f"--- Environment configured. HF_HOME is now: {hf_home} ---")

# --- (Sections 0, 1, 2, and 3 are unchanged) ---
# ==============================================================================
# 0. ROBUST ENVIRONMENT & CACHE SETUP
# ==============================================================================
def setup_environment():
    """Loads environment variables from .env and sets up cache directories."""
    from dotenv import load_dotenv
    print("--- Setting up environment ---")
    load_dotenv()
    hf_home, tmp_dir = os.getenv("HF_HOME"), os.getenv("TMPDIR")
    if not hf_home or not tmp_dir: raise ValueError("FATAL: HF_HOME and TMPDIR must be set in your .env file.")
    os.environ.update({
        "HF_HOME": hf_home, "HUGGINGFACE_HUB_CACHE": os.path.join(hf_home, "hub"),
        "TRANSFORMERS_CACHE": os.path.join(hf_home, "models"), "HF_DATASETS_CACHE": os.path.join(hf_home, "datasets"),
        "TMPDIR": tmp_dir, "HF_HUB_DISABLE_XET_STORAGE": "1",
    })
    for path in [os.path.join(hf_home, "hub"), os.path.join(hf_home, "models"), os.path.join(hf_home, "datasets"), tmp_dir]:
        os.makedirs(path, exist_ok=True)
    print(f"HF_HOME set to: {os.getenv('HF_HOME')}")
    print("Environment setup complete.")

setup_environment()

# ==============================================================================
# 1. CENTRAL CONFIGURATION & SETUP
# ==============================================================================
load_dotenv()
RANDOM_SEED = 42
N_SAMPLES_MMLU, N_SAMPLES_GAIA, N_SAMPLES_BBQ = 50, 53, 50
TARGET_PERSONALITIES = ["extraversion", "agreeableness", "neuroticism", "openness", "conscientiousness"]
STRATEGIC_MMLU_SUBJECTS = ["high_school_psychology", "abstract_algebra", "college_physics", "high_school_us_history", "logical_fallacies", "professional_law", "moral_scenarios"]
# NEW: Define the path to your local, pre-filtered BBQ dataset
BBQ_LOCAL_CSV_PATH = "/cs/student/projects3/aisd/2024/ghanda/bbq_ambiguous_with_metadata.csv"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
device_map = {"": 0}
BATCH_SIZE = 8

CONFIGS = {
    "gemma2": {
        "model_id": "google/gemma-2-2b-it",
        "peft": {"adapter_dir": "peft_gemma2_personality"},
        "steering": {
            "vector_dir": "persona_vectors_cache_big_five",
            "filename_overrides": {"openness": "openness_v3_combined_gemma-2-2b-it.pt"},
            "settings": {
               "extraversion":      {"layer": 15, "strength": 200.0}, "agreeableness":     {"layer": 10, "strength": 100.0},
               "neuroticism":       {"layer": 15,  "strength": 200.0},"openness":          {"layer": 15, "strength": 110.0},
               "conscientiousness": {"layer": 15, "strength": 250.0},
           }
        },
        "prompting": {"persona_template": "You are an expert assistant who embodies the personality trait of {personality}. Your task is to solve the following problem."}
    },
    "llama3": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "peft": {"adapter_dir": "peft_llama3_final"},
        "steering": {
            "vector_dir": "persona_vectors_cache_big_five",
            "filename_overrides": {},
            "settings": {
                # You MUST fill in your actual calibrated Llama-3 values here
            }
        },
        "prompting": {"persona_template": "You are an expert assistant who embodies the personality trait of {personality}. Your task is to solve the following problem."}
    }
}

print("--- Loading personality examples from 'holistic-ai/personality_manipulation' dataset ---")
try:
    # Load the training split of the dataset from the Hub
    personality_dataset = load_dataset("holistic-ai/personality_manipulation", split="train")
    df_personality_train = personality_dataset.to_pandas()
    
    # Sort for consistent example order in the prompt
    sorted_personalities = sorted(df_personality_train['Target Personality'].unique().tolist())
    
    personality_examples = {}
    for trait in sorted_personalities:
        trait_df = df_personality_train[df_personality_train['Target Personality'] == trait]
        # Use 2 examples per trait to keep prompt length manageable
        personality_examples[trait] = list(zip(trait_df['Question'], trait_df['Answer']))[:2]
    
    print(f"Loaded {len(personality_examples)} sets of personality examples successfully.")

except Exception as e:
    print(f"FATAL WARNING: Could not load personality examples from the Hub. Full-context prompting will fail. Error: {e}")
    personality_examples = None

# ==============================================================================
# 2. PROMPT HELPER
# ==============================================================================
def create_full_context_prompt(target_personality, all_examples):
    """
    Creates the context block showing examples of all 5 personality traits.
    This part of the prompt teaches the model to differentiate between traits.
    """
    if not all_examples:
        return "ERROR: Personality examples not loaded."

    full_context_str = ""
    for trait, examples in all_examples.items():
        full_context_str += f"--- EXAMPLES of '{trait}' personality ---\n"
        example_texts = [f"Question: {q}\nAnswer: {a}" for q, a in examples]
        full_context_str += "\n\n".join(example_texts)
        full_context_str += "\n\n"
    
    return (
        "You will be shown examples of five different personality traits to help you understand the differences between them.\n\n"
        f"{full_context_str}"
        "--- YOUR TASK ---\n"
        "Now that you have seen examples of all five personalities, your task is to solve the following problem. "
        f"You must adopt the '{target_personality}' personality strongly and clearly in your response."
    )

def create_benchmark_prompt(question, choices, benchmark_type, tokenizer, method_config=None):
    """
    Main prompt creation function. Now integrates full-context prompting.
    """
    method_config = method_config or {}
    method = method_config.get("method")
    personality = method_config.get("personality")

    # --- LOGIC CHANGE IS HERE ---
    if method == "prompting" and personality != "Baseline":
        # For the "prompting" method, we build the full-context instruction header.
        instruction_header = create_full_context_prompt(personality, personality_examples)
    else:
        # For all other methods (steering, peft, baseline), use the simple header.
        instruction_header = "You are an expert assistant. Your task is to solve the following problem."
    
    # The rest of the prompt structure remains the same
    instruction_body = (
        "First, you must reason through the problem step-by-step in a deliberate manner. "
        "After your reasoning, you MUST conclude with the final answer on a new line in the specified format."
    )
    full_instruction = f"{instruction_header}\n\n{instruction_body}"

    if benchmark_type == "MMLU":
        choice_str, final_fmt = "\n".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)]), "Final Answer: <LETTER>"
    elif benchmark_type == "BBQ":
        choice_str, final_fmt = "\n".join([f"({i}) {c}" for i, c in enumerate(choices)]), "Final Answer: <INDEX>"
    else: # GAIA
        choice_str, final_fmt = "", "Final Answer: <ANSWER>"
    
    task_content = (
        f"{question}\n\n{choice_str}\n\n"
        "Instructions:\n"
        "1. Reason step-by-step.\n"
        f"2. Conclude with the answer in the format: {{final_fmt}}"
    )
    
    full_user_prompt = f"{full_instruction}\n\n---\n\n{task_content}"
    messages = [{"role": "user", "content": full_user_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ==============================================================================
# 3. CORE GENERATION & BENCHMARK RUNNERS (GENERATE ONLY)
# ==============================================================================
def generate_batched_responses(model, tokenizer, prompts, max_new_tokens=512, method_config=None):
    method_config = method_config or {}
    hook_handle = None
    try:
        method, personality = method_config.get("method"), method_config.get("personality")
        if method == "steering" and personality != "Baseline":
            settings, vectors = method_config.get("settings", {}), method_config.get("vectors", {})
            if personality in settings and personality in vectors:
                steer_layer, steer_strength = settings[personality]["layer"], settings[personality]["strength"]
                steer_vector = vectors[personality].get(steer_layer)
                if steer_vector is not None:
                    gpu_steer_vector = steer_vector.to(model.device)
                    def hook(module, input, output): return output + gpu_steer_vector.to(output.dtype) * steer_strength
                    target_module = model.model.layers[steer_layer].post_attention_layernorm
                    hook_handle = target_module.register_forward_hook(hook)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        generation_params = {'do_sample': True, 'temperature': 0.6, 'top_p': 0.9, 'max_new_tokens': max_new_tokens, 'pad_token_id': tokenizer.eos_token_id}
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_params)
        responses = []
        prompt_lengths = inputs['input_ids'].shape[1]
        for i in range(len(prompts)):
            response_ids = outputs[i][prompt_lengths:]
            responses.append(tokenizer.decode(response_ids, skip_special_tokens=True).strip())
        return responses
    finally:
        if hook_handle: hook_handle.remove()

def run_generation_for_benchmark(benchmark_name, dataset, model, tokenizer, method_config):
    prompts = []
    if benchmark_name == "MMLU":
        for item in dataset: prompts.append(create_benchmark_prompt(item['question'], item['choices'], "MMLU", tokenizer, method_config))
    elif benchmark_name == "GAIA":
        for item in dataset:
            prompt_text = item['Question'] + (f"\n\n--- Document ---\n{item['file_content']}" if item.get('file_content') else "")
            prompts.append(create_benchmark_prompt(prompt_text, [], "GAIA", tokenizer, method_config))
    elif benchmark_name == "BBQ":
        for item in dataset: prompts.append(create_benchmark_prompt(f"{item['context']}\n{item['question']}", [item['ans0'], item['ans1'], item['ans2']], "BBQ", tokenizer, method_config))

    all_responses = []
    max_tokens = 256 if benchmark_name == "BBQ" else 512
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"    Generating for {benchmark_name[:10]:<10}"):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        responses = generate_batched_responses(model, tokenizer, batch_prompts, max_tokens, method_config)
        all_responses.extend(responses)

    output_data = []
    for i, item in enumerate(dataset):
        record = dict(item)
        record['model_response'] = all_responses[i]
        output_data.append(record)
    return output_data

# ==============================================================================
# 4. MAIN ORCHESTRATION SCRIPT (GENERATE & SAVE)
# ==============================================================================
def main(args):
    global BATCH_SIZE
    if args.batch_size: BATCH_SIZE = args.batch_size
    print(f"--- Using Batch Size: {BATCH_SIZE} ---")
    
    logging.set_verbosity_error()
    MODEL_CONFIG, METHOD_CONFIG = CONFIGS[args.model], CONFIGS[args.model][args.method]

    print(f"\n--- Loading base model: {MODEL_CONFIG['model_id']} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_id'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIG['model_id'], quantization_config=bnb_config, device_map=device_map, attn_implementation="sdpa", torch_dtype=torch.bfloat16, cache_dir="/cs/student/projects3/aisd/2024/ghanda/hf_cache",)
    
    model_for_eval = base_model
    # ... (PEFT and Steering loading logic is unchanged) ...
    if args.method == 'peft':
        print(f"--- Loading PEFT adapters from: {METHOD_CONFIG['adapter_dir']} ---")
        peft_model, loaded = base_model, False
        for trait in TARGET_PERSONALITIES:
            adapter_path = os.path.join(METHOD_CONFIG['adapter_dir'], trait)
            if not os.path.exists(adapter_path): print(f"Warning: PEFT Adapter for '{trait}' not found. Skipping."); continue
            if not loaded: peft_model, loaded = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=trait), True
            else: peft_model.load_adapter(adapter_path, adapter_name=trait)
            print(f"Loaded PEFT adapter: '{trait}'")
        if not loaded: raise RuntimeError("FATAL: No PEFT adapters were found.")
        model_for_eval = peft_model
    elif args.method == 'steering':
        print(f"--- Loading Steering vectors from: {METHOD_CONFIG['vector_dir']} ---")
        vectors_by_trait, overrides = {}, METHOD_CONFIG.get("filename_overrides", {})
        model_filename_id = MODEL_CONFIG['model_id'].split('/')[-1]
        for trait in TARGET_PERSONALITIES:
            filename = overrides[trait] if trait in overrides else f"{trait}_{model_filename_id}.pt"
            vec_path = os.path.join(METHOD_CONFIG['vector_dir'], filename)
            if os.path.exists(vec_path):
                vectors_by_trait[trait] = torch.load(vec_path)
                print(f"Loaded steering vector for '{trait}' from '{filename}'")
            else: print(f"Warning: Steering vector for '{trait}' not found at '{vec_path}'. Skipping.")
        METHOD_CONFIG["vectors"] = vectors_by_trait

    print(f"\n--- Preparing benchmark: {args.benchmark.upper()} ---")
    all_benchmark_data = []
    if args.benchmark.upper() == "MMLU":
        for subject in STRATEGIC_MMLU_SUBJECTS:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True).shuffle(seed=RANDOM_SEED).select(range(N_SAMPLES_MMLU))
                all_benchmark_data.extend([{'benchmark_type': 'MMLU', 'subject': subject, **item} for item in ds])
            except Exception as e: print(f"    [Warning] Could not load MMLU '{subject}'. Skipping. Error: {e}")
    elif args.benchmark.upper() == "GAIA":
        try:
            ds = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation", trust_remote_code=True).shuffle(seed=RANDOM_SEED).select(range(N_SAMPLES_GAIA))
            all_benchmark_data.extend([{'benchmark_type': 'GAIA', **item} for item in ds])
        except Exception as e: print(f"    [Warning] Could not load GAIA. Skipping. Error: {e}")
    
    # THE FIX IS HERE: This block now loads your local CSV for the BBQ benchmark.
    elif args.benchmark.upper() == "BBQ":
        try:
            print(f"--- Loading local ambiguous BBQ data from: {BBQ_LOCAL_CSV_PATH} ---")
            df_bbq_full = pd.read_csv(BBQ_LOCAL_CSV_PATH)
            
            # Ensure required columns exist
            required_cols = ['category_x', 'context', 'question', 'ans0', 'ans1', 'ans2', 'target_loc']
            if not all(col in df_bbq_full.columns for col in required_cols):
                raise KeyError(f"Local BBQ CSV is missing one of the required columns: {required_cols}")

            # Get unique categories from your local file
            local_bbq_categories = df_bbq_full['category_x'].unique()
            print(f"Found categories in local file: {list(local_bbq_categories)}")

            for category in local_bbq_categories:
                df_category = df_bbq_full[df_bbq_full['category_x'] == category]
                # Determine sample size, ensuring we don't sample more than available unless we allow replacement
                num_to_sample = min(N_SAMPLES_BBQ, len(df_category))
                df_sampled = df_category.sample(n=num_to_sample, random_state=RANDOM_SEED)
                
                # Convert the sampled DataFrame to a list of dicts for consistency
                for item in df_sampled.to_dict('records'):
                     # Remap the category column name for downstream compatibility in the scoring script
                    item['category'] = item['category_x']
                    all_benchmark_data.append({'benchmark_type': 'BBQ', **item})

        except FileNotFoundError:
            print(f"    [FATAL] Local BBQ CSV not found at '{BBQ_LOCAL_CSV_PATH}'. Aborting.")
        except Exception as e:
            print(f"    [FATAL] Could not load local BBQ data. Aborting. Error: {e}")

    if not all_benchmark_data: print("FATAL: No benchmark data loaded. Exiting."); return
    benchmark_dataset = Dataset.from_pandas(pd.DataFrame(all_benchmark_data))

    full_raw_results = []
    for personality in ["Baseline"] + TARGET_PERSONALITIES:
        print(f"\n{'='*20} GENERATING FOR: {personality.upper()} on {args.benchmark.upper()} (Method: {args.method}) {'='*20}")
        current_config = {"method": args.method, "personality": personality, **METHOD_CONFIG}
        if args.method == 'peft' and personality != 'Baseline': model_for_eval.set_adapter(personality)
        
        generated_data = run_generation_for_benchmark(args.benchmark.upper(), benchmark_dataset, model_for_eval, tokenizer, current_config)
        for record in generated_data:
            record['model_personality'] = personality
        full_raw_results.extend(generated_data)

    output_filename = f"raw_outputs_{args.model}_{args.benchmark}_{args.method}.jsonl"
    pd.DataFrame(full_raw_results).to_json(output_filename, orient='records', lines=True)
    print(f"\n\n{'='*80}\n✅ Generation Complete!\nRaw outputs saved to '{output_filename}'\n{'='*80}")
    print(f"Next step: run 'python evaluate_score.py {output_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAGE 1: Generate model outputs for benchmarks.")
    parser.add_argument("--model", type=str, required=True, choices=["gemma2", "llama3"], help="The base model to evaluate.")
    parser.add_argument("--benchmark", type=str, required=True, choices=["MMLU", "GAIA", "BBQ"], help="The benchmark to run.")
    parser.add_argument("--method", type=str, required=True, choices=["peft", "steering", "prompting"], help="The method for applying personality.")
    parser.add_argument("--batch_size", type=int, help="Override the default batch size for generation.")
    args = parser.parse_args()
    main(args)