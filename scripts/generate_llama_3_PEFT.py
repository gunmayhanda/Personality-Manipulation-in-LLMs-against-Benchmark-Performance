"""
This script performs a fine-tuning and evaluation experiment to manipulate the personality
of a large language model. It trains separate PEFT adapters for different personality
traits, generates responses from the base and fine-tuned models, and performs a
rigorous statistical analysis of the results.

The process includes:
1. Training LoRA adapters for specified personality traits on a base model.
2. Generating responses to a test set of questions from both the base model and
   each of the fine-tuned adapters.
3. Using a judge model (via Azure OpenAI) to assess the quality of each generated response.
4. Using a personality classifier to determine the personality trait of valid responses.
5. Performing statistical analysis to compare the personality alignment of the PEFT models
   against the baseline.
"""
import os
import argparse
import logging
import time
from pathlib import Path
from functools import partial
from typing import Dict, Any, List

import torch
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

# Hugging Face imports
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    TrainingArguments,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import login

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# These can be overridden by command-line arguments
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_OUTPUT_DIR = "peft_llama3_results"
DEFAULT_DATASET = "holistic-ai/personality_manipulation"

# --- Quantization Config ---
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = False
DEVICE_MAP = {"": 0}

# --- PEFT & Training Hyperparameters ---
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
OPTIMIZER = "paged_adamw_8bit"
LR_SCHEDULER = "cosine"
MAX_SEQ_LENGTH = 512
LOGGING_STEPS = 25

# --- Evaluation Parameters ---
N_GENERATIONS_PER_CONDITION = 1
RANDOM_SEED = 42
ALPHA_LEVEL = 0.05


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_llama3_prompt(sample: Dict[str, str], tokenizer: LlamaTokenizer, for_training: bool) -> Dict[str, str]:
    """Creates a Llama-3 formatted prompt for either training or inference."""
    if for_training:
        messages = [
            {"role": "user", "content": sample.get('Question')},
            {"role": "assistant", "content": sample.get('Answer')},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": prompt}
    else:
        messages = [{"role": "user", "content": sample.get('Question')}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def judge_response_quality(client: AzureOpenAI, deployment: str, question: str, response: str) -> str:
    """Uses an external LLM to judge the quality of a generated response."""
    if not isinstance(response, str) or not response.strip():
        return "EMPTY"

    system_prompt = (
        "You are a strict response quality judge. Your task is to determine if a response "
        "is valid or if it fails in a specific way. You must respond with ONLY ONE of the "
        "following single-word categories:\n"
        "- VALID: The response is a coherent, on-topic answer to the question.\n"
        "- REFUSAL: The response explicitly refuses to answer (e.g., 'As an AI...').\n"
        "- GIBBERISH: The response is nonsensical, repetitive, or unreadable.\n"
        "- OFF_TOPIC: The response is coherent but does not address the question asked.\n"
    )
    user_prompt = f"Question:\n---\n{question}\n---\n\nResponse to Judge:\n---\n{response}\n---\n\nCategory:"

    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=10, n=1,
        )
        judge_response = completion.choices[0].message.content.strip().upper()
        if judge_response in ["VALID", "REFUSAL", "GIBBERISH", "OFF_TOPIC"]:
            return judge_response
        return "JUDGE_FORMAT_ERROR"
    except Exception as e:
        logger.error(f"Azure API Error during quality judgment: {e}")
        return "JUDGE_API_ERROR"


def bootstrap_ci(data: np.ndarray, stat_func=np.mean, n_bootstrap=1000, confidence=0.95) -> tuple[float, float]:
    """Calculates a bootstrap confidence interval for a given statistic."""
    if len(data) == 0:
        return (0.0, 0.0)
    bootstrap_stats = [stat_func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    alpha = 1 - confidence
    return np.percentile(bootstrap_stats, 100 * alpha / 2), np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))


# ==============================================================================
# EVALUATION CLASS
# ==============================================================================

class RigorousEvaluator:
    """Manages the process of generating, judging, and classifying model responses."""

    def __init__(self, tokenizer: LlamaTokenizer, personality_classifier,
                 azure_client: AzureOpenAI, azure_deployment: str, seed: int):
        self.tokenizer = tokenizer
        self.personality_classifier = personality_classifier
        self.azure_client = azure_client
        self.azure_deployment = azure_deployment
        self.seed = seed
        self.generation_params = {'do_sample': True, 'temperature': 0.7, 'top_p': 0.9}
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_responses(self, model: LlamaForCausalLM, questions_df: pd.DataFrame,
                           condition_name: str, n_runs: int) -> List[Dict[str, Any]]:
        """Generates responses for a given model and condition across multiple runs."""
        all_results = []
        for run_id in range(n_runs):
            logger.info(f"  Starting generation run {run_id + 1}/{n_runs} for condition '{condition_name}'...")
            desc = f"Run {run_id+1} ({condition_name})"
            for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc=desc, leave=False):
                question = row.get('Question')
                if not isinstance(question, str) or not question.strip():
                    continue

                torch.manual_seed(self.seed + run_id * 1000 + row.name)
                inference_prompt = create_llama3_prompt({"Question": question}, self.tokenizer, for_training=False)
                
                inputs = self.tokenizer(inference_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=150, pad_token_id=self.tokenizer.eos_token_id, **self.generation_params
                    )
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                
                all_results.append({
                    'condition': condition_name, 'run_id': run_id, 'question_id': row.name,
                    'question': question, 'target_personality': row.get('Target Personality'),
                    'llm_raw_response': response
                })
        return all_results

    def analyze_responses(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Judges response quality and classifies personality for valid responses."""
        logger.info("Step 1: Judging response quality with Azure OpenAI...")
        if not self.azure_client:
            logger.error("Azure client not available. Skipping quality control.")
            for r in results_list:
                r['quality_judgment'] = 'JUDGE_SKIPPED'
        else:
            for result in tqdm(results_list, desc="Quality Judging"):
                result['quality_judgment'] = judge_response_quality(
                    client=self.azure_client, deployment=self.azure_deployment,
                    question=result['question'], response=result['llm_raw_response']
                )

        logger.info("Step 2: Classifying personality for VALID responses...")
        valid_responses = [r for r in results_list if r.get('quality_judgment') == 'VALID']
        if not valid_responses:
            logger.warning("No valid responses found to classify.")
        else:
            try:
                responses_to_classify = [r['llm_raw_response'] for r in valid_responses]
                classifier_output = self.personality_classifier(responses_to_classify, batch_size=32, truncation=True)
                for i, result in enumerate(valid_responses):
                    result['predicted_trait'] = classifier_output[i]['label']
                    result['trait_confidence'] = classifier_output[i]['score']
            except Exception as e:
                logger.warning(f"Personality classification failed: {e}. Marking as 'unknown'.")
                for r in valid_responses:
                    r.update({'predicted_trait': 'unknown', 'trait_confidence': 0.0})

        for r in results_list:
            r.setdefault('predicted_trait', 'unclassified')
            r.setdefault('trait_confidence', 0.0)

        return pd.DataFrame(results_list)


# ==============================================================================
# MAIN SCRIPT FUNCTIONS
# ==============================================================================

def setup_environment() -> AzureOpenAI | None:
    """Loads environment variables and initializes clients."""
    load_dotenv()
    logger.info("Loaded environment variables from .env file.")

    if hf_token := os.getenv("HF_TOKEN"):
        login(token=hf_token)
        logger.info("Successfully logged into Hugging Face.")
    else:
        logger.warning("Hugging Face token not found. Skipping login.")

    try:
        azure_client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY"),
        )
        logger.info("Azure OpenAI client for quality control initialized successfully.")
        return azure_client
    except Exception as e:
        logger.error(f"Could not initialize Azure OpenAI client: {e}")
        return None


def train_adapters(args: argparse.Namespace, tokenizer: LlamaTokenizer, target_personalities: List[str]):
    """Trains a PEFT adapter for each specified personality trait."""
    logger.info("--- PART 1: Loading/Training Models ---")
    
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )

    dataset = load_dataset(args.dataset_name)
    full_train_df = dataset['train'].to_pandas()

    for trait in target_personalities:
        trait_key = trait.lower().replace(" ", "_")
        output_dir = args.output_dir / trait_key
        
        if output_dir.exists() and any(output_dir.iterdir()):
            logger.info(f"Adapter for '{trait}' already exists at '{output_dir}'. Skipping training.")
            continue

        logger.info(f"\n{'='*20} TRAINING: {trait.upper()} {'='*20}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map=DEVICE_MAP,
            torch_dtype=compute_dtype,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        
        trait_df = full_train_df[full_train_df['Target Personality'] == trait]
        prompt_map_fn = partial(create_llama3_prompt, tokenizer=tokenizer, for_training=True)
        train_dataset = Dataset.from_pandas(trait_df).map(prompt_map_fn, load_from_cache_file=False)

        peft_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES
        )
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            optim=OPTIMIZER,
            learning_rate=args.learning_rate,
            lr_scheduler_type=LR_SCHEDULER,
            logging_steps=LOGGING_STEPS,
            save_strategy="epoch",
            bf16=True, fp16=False, group_by_length=True,
            report_to="tensorboard", save_total_limit=1,
            seed=RANDOM_SEED
        )
        trainer = SFTTrainer(
            model=model, args=training_args, train_dataset=train_dataset,
            peft_config=peft_config,
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        
        del model, trainer
        torch.cuda.empty_cache()
        logger.info(f"Finished training for {trait.upper()}. Model saved to {output_dir}")


def run_evaluation(args: argparse.Namespace, tokenizer: LlamaTokenizer, target_personalities: List[str], azure_client: AzureOpenAI) -> pd.DataFrame:
    """Runs the full evaluation pipeline for baseline and all PEFT models."""
    logger.info("\n" + "="*80)
    logger.info(f"STARTING RIGOROUS EVALUATION ON '{args.dataset_name}' TEST SET")
    logger.info("="*80)

    df_test = load_dataset(args.dataset_name, split="test").to_pandas()
    test_subset = df_test.sample(n=min(len(df_test), args.n_samples), random_state=RANDOM_SEED)
    logger.info(f"Evaluating on {len(test_subset)} questions with {N_GENERATIONS_PER_CONDITION} run(s) per condition.")

    personality_classifier = pipeline("text-classification", model="holistic-ai/personality_classifier")
    evaluator = RigorousEvaluator(
        tokenizer, personality_classifier, azure_client,
        os.getenv("AZURE_OPENAI_DEPLOYMENT"), RANDOM_SEED
    )
    
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT, bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=USE_NESTED_QUANT
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map=DEVICE_MAP, attn_implementation="sdpa"
    )
    
    all_results = []
    logger.info("\n--- Generating BASELINE responses ---")
    all_results.extend(evaluator.generate_responses(base_model, test_subset, "Baseline", N_GENERATIONS_PER_CONDITION))

    logger.info("\n--- Generating PEFT responses ---")
    try:
        # Load the first adapter to create the PeftModel
        first_trait_key = target_personalities[0].lower().replace(" ", "_")
        peft_model = PeftModel.from_pretrained(base_model, str(args.output_dir / first_trait_key), adapter_name=first_trait_key)
        
        # Load remaining adapters
        for trait in target_personalities[1:]:
            adapter_key = trait.lower().replace(" ", "_")
            peft_model.load_adapter(str(args.output_dir / adapter_key), adapter_name=adapter_key)
        
        # Evaluate each adapter by activating it
        for trait in target_personalities:
            adapter_key = trait.lower().replace(" ", "_")
            logger.info(f"\n--- Evaluating {trait.upper()} Adapter ---")
            peft_model.set_adapter(adapter_key)
            all_results.extend(evaluator.generate_responses(peft_model, test_subset, f"PEFT_{trait}", N_GENERATIONS_PER_CONDITION))
            
    except Exception as e:
        logger.critical(f"Failed to load or evaluate PEFT adapters: {e}", exc_info=True)
        # Continue with baseline-only analysis if adapters fail
    
    del base_model
    if 'peft_model' in locals():
        del peft_model
    torch.cuda.empty_cache()

    logger.info("\n--- ANALYZING ALL GENERATED RESPONSES ---")
    df_results = evaluator.analyze_responses(all_results)
    return df_results


def analyze_and_save_results(df_results: pd.DataFrame, target_personalities: List[str], args: argparse.Namespace):
    """Performs final statistical analysis and saves results to CSV."""
    logger.info("\n--- Quality Control Summary ---")
    quality_summary = df_results.groupby('condition')['quality_judgment'].value_counts().unstack(fill_value=0)
    logger.info("Number of responses per quality category:\n" + quality_summary.to_string())

    df_valid = df_results[df_results['quality_judgment'] == 'VALID'].copy()
    logger.info(f"\nTotal responses generated: {len(df_results)}")
    logger.info(f"Total VALID responses for analysis: {len(df_valid)}")
    
    if len(df_valid) == 0:
        logger.warning("No valid responses were generated. Skipping statistical analysis.")
    else:
        logger.info("\n" + "="*80)
        logger.info("STATISTICAL ANALYSIS: PERSONALITY ALIGNMENT (ON VALID RESPONSES)")
        logger.info("="*80)

        df_valid['is_aligned'] = (df_valid['predicted_trait'] == df_valid['target_personality'])
        baseline_df = df_valid[df_valid['condition'] == 'Baseline']
        peft_df = df_valid[df_valid['condition'].str.startswith('PEFT_')]

        if len(baseline_df) == 0:
            logger.warning("No valid baseline data to perform analysis.")
            return

        baseline_rate = baseline_df['is_aligned'].mean()
        baseline_ci = bootstrap_ci(baseline_df['is_aligned'].astype(int))
        logger.info(f"\n--- Baseline Model ---")
        logger.info(f"Overall Personality Alignment Rate: {baseline_rate:.2%} (95% CI: {baseline_ci[0]:.2%} - {baseline_ci[1]:.2%})")
        
        p_values = []
        peft_results = {}
        traits_for_stats = []

        for personality in target_personalities:
            peft_condition_name = f'PEFT_{personality}'
            peft_subset = peft_df[(peft_df['condition'] == peft_condition_name) & (peft_df['target_personality'] == personality)]
            baseline_subset = baseline_df[baseline_df['target_personality'] == personality]

            if len(peft_subset) < 2 or len(baseline_subset) < 2:
                 logger.warning(f"Skipping stats for '{personality}' due to insufficient data for comparison.")
                 continue
            
            traits_for_stats.append(personality)
            peft_scores = peft_subset['is_aligned'].astype(int)
            baseline_scores = baseline_subset['is_aligned'].astype(int)
            _, p_val = stats.mannwhitneyu(peft_scores, baseline_scores, alternative='two-sided')
            p_values.append(p_val)
            peft_results[personality] = {
                'peft_rate': peft_scores.mean(), 'baseline_rate': baseline_scores.mean(),
                'n': len(peft_scores)
            }
        
        if not p_values:
            logger.warning("No PEFT models could be statistically compared.")
        else:
            corrected_p_values = multipletests(p_values, method='bonferroni')[1]
            logger.info("\n--- PEFT vs Baseline Comparison (on target-specific questions) ---")
            logger.info("(P-values are Bonferroni-corrected for multiple comparisons)")
            for i, personality in enumerate(traits_for_stats):
                res = peft_results[personality]
                p_corr = corrected_p_values[i]
                significance = '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'ns'
                logger.info(f"\n--- {personality.upper()} Model ---")
                logger.info(f"Alignment on '{personality}' questions: PEFT: {res['peft_rate']:.2%} vs Baseline: {res['baseline_rate']:.2%}")
                logger.info(f"  p-value (corrected): {p_corr:.4f} ({significance}) | n = {res['n']}")

    results_filename = args.output_dir / "full_experiment_results.csv"
    df_results.to_csv(results_filename, index=False)
    logger.info(f"\nFull, detailed results saved to '{results_filename}'")


def main():
    parser = argparse.ArgumentParser(description="Run a personality manipulation experiment on an LLM.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Base model to fine-tune and evaluate.")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET, help="Dataset for training and evaluation.")
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Directory to save adapters and results.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training.")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of test samples for evaluation.")
    parser.add_argument("--skip_training", action="store_true", help="Skip the training phase and proceed to evaluation.")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("--- Initializing Personality Evaluation Experiment ---")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    azure_client = setup_environment()
    if not azure_client:
        logger.critical("Azure client setup failed. Evaluation quality control will be skipped or fail.")
        # Depending on strictness, one might exit here: `return`
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    try:
        dataset = load_dataset(args.dataset_name, split="train")
        target_personalities = list(dataset.to_pandas()['Target Personality'].unique())
        logger.info(f"Found target personalities: {target_personalities}")
    except Exception as e:
        logger.critical(f"Could not load personality dataset '{args.dataset_name}': {e}", exc_info=True)
        return

    if not args.skip_training:
        train_adapters(args, tokenizer, target_personalities)
    else:
        logger.info("Skipping training phase as requested.")

    df_results = run_evaluation(args, tokenizer, target_personalities, azure_client)
    
    if not df_results.empty:
        analyze_and_save_results(df_results, target_personalities, args)

    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info(f"EVALUATION COMPLETE. Total runtime: {total_time / 60:.2f} minutes.")
    logger.info("="*80)


if __name__ == "__main__":
    main()