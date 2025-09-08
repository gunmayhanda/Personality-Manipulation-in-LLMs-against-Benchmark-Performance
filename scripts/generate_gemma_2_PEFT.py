"""
This script conducts an experiment to fine-tune and evaluate the personality traits of
the Gemma-2 model. It trains separate PEFT LoRA adapters for various personality
types and then compares their alignment with the target personality against a
baseline model.

The workflow is as follows:
1.  Train LoRA adapters for each target personality using the specified dataset.
2.  Generate responses from both the base model and each fine-tuned adapter on a
    held-out test set.
3.  Classify the personality of each generated response using a pre-trained
    text-classification model.
4.  Perform a statistical analysis (Mann-Whitney U test) to determine if the
    fine-tuned models show a significant shift towards their target personality
    compared to the baseline.
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
from dotenv import load_dotenv
from scipy import stats

# Hugging Face imports
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    TrainingArguments,
    GemmaForCausalLM,
    GemmaTokenizer,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import login
from tqdm.auto import tqdm

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
DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"
DEFAULT_OUTPUT_DIR = "peft_gemma2_personality"
DEFAULT_DATASET = "holistic-ai/personality_manipulation"
DEFAULT_CLASSIFIER = "holistic-ai/personality_classifier"

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
RANDOM_SEED = 42


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_gemma_prompt(sample: Dict[str, str], tokenizer: GemmaTokenizer, for_training: bool) -> Dict[str, str] | str:
    """Creates a Gemma-formatted prompt for either training or inference."""
    messages = [{"role": "user", "content": sample['Question']}]
    if for_training:
        messages.append({"role": "assistant", "content": sample['Answer']})
        # For SFTTrainer, return a dictionary with a "text" key
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    else:
        # For inference, return the prompt string directly
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load_model_and_tokenizer(model_name: str) -> tuple[GemmaForCausalLM, GemmaTokenizer]:
    """Loads the quantized model and tokenizer."""
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )
    
    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=DEVICE_MAP,
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


# ==============================================================================
# EVALUATION CLASS
# ==============================================================================

class PersonalityEvaluator:
    """Manages the generation and personality classification of model responses."""

    def __init__(self, tokenizer: GemmaTokenizer, classifier_pipeline, seed: int):
        self.tokenizer = tokenizer
        self.classifier = classifier_pipeline
        self.seed = seed
        self.generation_params = {
            'do_sample': True, 'temperature': 0.7, 'top_p': 0.9,
            'max_new_tokens': 150, 'pad_token_id': self.tokenizer.eos_token_id,
        }
        np.random.seed(seed)
        torch.manual_seed(seed)

    def evaluate_condition(self, model: GemmaForCausalLM, questions_df: pd.DataFrame, condition_name: str) -> pd.DataFrame:
        """Generates and classifies responses for a given model and condition."""
        logger.info(f"Evaluating condition '{condition_name}' on {len(questions_df)} questions...")
        
        results = []
        for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc=f"Generating for {condition_name}"):
            torch.manual_seed(self.seed + idx)
            prompt = create_gemma_prompt({'Question': row['Question']}, self.tokenizer, for_training=False)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **self.generation_params)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            results.append({
                'condition': condition_name,
                'question': row['Question'],
                'target_personality': row['Target Personality'],
                'llm_raw_response': response,
            })

        # Batch classify responses for efficiency
        logger.info(f"  Classifying {len(results)} responses...")
        responses_to_classify = [r['llm_raw_response'] for r in results]
        try:
            classifier_output = self.classifier(responses_to_classify, batch_size=32, truncation=True)
            for i, result in enumerate(results):
                result['predicted_trait'] = classifier_output[i]['label']
                result['trait_confidence'] = classifier_output[i]['score']
        except Exception as e:
            logger.error(f"Personality classification failed: {e}", exc_info=True)
            for result in results:
                result.update({'predicted_trait': 'classification_error', 'trait_confidence': 0.0})

        return pd.DataFrame(results)


# ==============================================================================
# MAIN SCRIPT FUNCTIONS
# ==============================================================================

def train_adapters(args: argparse.Namespace, target_personalities: List[str]):
    """Trains a PEFT adapter for each personality trait."""
    logger.info("--- PART 1: Training Personality Adapters ---")

    # Load model and tokenizer once for all training runs
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    dataset = load_dataset(args.dataset_name)
    full_train_df = dataset['train'].to_pandas()

    for trait in target_personalities:
        output_dir = args.output_dir / trait
        
        if output_dir.exists() and any(output_dir.iterdir()):
            logger.info(f"Adapter for '{trait}' found at '{output_dir}'. Skipping training.")
            continue

        logger.info(f"\n{'='*20} TRAINING: {trait.upper()} {'='*20}")
        model.config.use_cache = False
        
        trait_df = full_train_df[full_train_df['Target Personality'] == trait]
        prompt_map_fn = partial(create_gemma_prompt, tokenizer=tokenizer, for_training=True)
        train_dataset = Dataset.from_pandas(trait_df).map(
            prompt_map_fn,
            remove_columns=list(trait_df.columns)
        )

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
        logger.info(f"Finished training for {trait.upper()}. Adapter saved to {output_dir}")

    del model
    torch.cuda.empty_cache()


def run_evaluation(args: argparse.Namespace, target_personalities: List[str]) -> pd.DataFrame:
    """Runs the full evaluation pipeline for baseline and all PEFT models."""
    logger.info("\n" + "="*80)
    logger.info("PART 2: STARTING PERSONALITY ALIGNMENT EVALUATION")
    logger.info("="*80)

    eval_df = load_dataset(args.dataset_name, split="test").to_pandas()
    eval_subset = eval_df.sample(n=min(len(eval_df), args.n_samples), random_state=RANDOM_SEED)
    logger.info(f"Evaluating on {len(eval_subset)} questions from the test set.")

    # Load model and tokenizer
    base_model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Load classifier and evaluator
    classifier = pipeline("text-classification", model=args.classifier_name, device=0)
    evaluator = PersonalityEvaluator(tokenizer, classifier, RANDOM_SEED)
    
    all_results = []
    
    # --- Baseline Evaluation ---
    logger.info("\n--- Evaluating BASELINE Model ---")
    all_results.append(evaluator.evaluate_condition(base_model, eval_subset, "Baseline"))

    # --- PEFT Evaluation ---
    logger.info("\n--- Evaluating PEFT Adapters ---")
    try:
        # Load the first adapter to create the PeftModel object
        first_trait = target_personalities[0]
        peft_model = PeftModel.from_pretrained(base_model, str(args.output_dir / first_trait), adapter_name=first_trait)
        logger.info(f"Loaded initial adapter: '{first_trait}'")
        
        # Load all other adapters
        for trait in target_personalities[1:]:
            adapter_path = args.output_dir / trait
            if adapter_path.exists():
                peft_model.load_adapter(str(adapter_path), adapter_name=trait)
                logger.info(f"Loaded additional adapter: '{trait}'")
            else:
                logger.warning(f"Adapter for '{trait}' not found at '{adapter_path}'. Skipping.")

        # Evaluate each loaded adapter
        for trait in target_personalities:
            if trait not in peft_model.peft_config:
                continue
            logger.info(f"\n--- Activating Adapter: {trait.upper()} ---")
            peft_model.set_adapter(trait)
            all_results.append(evaluator.evaluate_condition(peft_model, eval_subset, f"PEFT_{trait}"))
    except Exception as e:
        logger.critical(f"Failed to load or evaluate PEFT adapters: {e}", exc_info=True)
        
    del base_model
    if 'peft_model' in locals():
        del peft_model
    torch.cuda.empty_cache()

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def analyze_and_report(df_results: pd.DataFrame, target_personalities: List[str], args: argparse.Namespace):
    """Performs statistical analysis and saves the final results."""
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL ANALYSIS: PERSONALITY ALIGNMENT")
    logger.info("="*80)

    baseline_df = df_results[df_results['condition'] == 'Baseline'].copy()
    peft_df = df_results[df_results['condition'].str.startswith('PEFT_')].copy()

    if baseline_df.empty or peft_df.empty:
        logger.error("Not enough data for statistical analysis. One or more conditions had no results.")
        return

    logger.info("\n--- Baseline Model Personality Distribution ---")
    baseline_dist = baseline_df['predicted_trait'].value_counts(normalize=True)
    logger.info("\n" + baseline_dist.to_string(float_format="{:.1%}".format))

    logger.info("\n--- PEFT vs. Baseline Alignment Comparison ---")
    logger.info("Alignment = %% of responses where predicted trait matches the PEFT adapter's target trait.\n")

    for personality in target_personalities:
        peft_condition_name = f'PEFT_{personality}'
        peft_subset = peft_df[peft_df['condition'] == peft_condition_name]
        if peft_subset.empty:
            continue

        # Alignment for PEFT is when its output matches its intended personality
        peft_aligned_scores = (peft_subset['predicted_trait'] == personality).astype(int)
        # Alignment for baseline is how often it *incidentally* produced that same personality
        baseline_aligned_scores = (baseline_df['predicted_trait'] == personality).astype(int)

        peft_rate = peft_aligned_scores.mean()
        baseline_rate = baseline_aligned_scores.mean()

        try:
            _, p_value = stats.mannwhitneyu(peft_aligned_scores, baseline_aligned_scores, alternative='two-sided')
            p_str = f"p={p_value:.4f}"
            if p_value < 0.001: p_str += " ***"
            elif p_value < 0.01: p_str += " **"
            elif p_value < 0.05: p_str += " *"
        except ValueError:
            p_str = "N/A (insufficient data)"

        logger.info(f"--- {personality.upper()} ---")
        logger.info(f"  PEFT Alignment:     {peft_rate:>7.1%}")
        logger.info(f"  Baseline Alignment: {baseline_rate:>7.1%}")
        logger.info(f"  Significance:       {p_str}")

    results_filename = args.output_dir / "gemma2_personality_results.csv"
    df_results.to_csv(results_filename, index=False)
    logger.info(f"\nFull evaluation results saved to '{results_filename}'")


def main():
    parser = argparse.ArgumentParser(description="Run a personality fine-tuning experiment on a Gemma-2 model.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Base model for the experiment.")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET, help="Dataset for training and evaluation.")
    parser.add_argument("--classifier_name", type=str, default=DEFAULT_CLASSIFIER, help="Personality classifier model.")
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Directory to save adapters and results.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training.")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of test samples for evaluation.")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and proceed directly to evaluation.")
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("--- Initializing Personality PEFT Experiment for Gemma-2 ---")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if token := os.getenv("HF_TOKEN"):
        login(token=token)

    try:
        dataset = load_dataset(args.dataset_name, split="train")
        target_personalities = list(dataset.to_pandas()['Target Personality'].unique())
        logger.info(f"Found target personalities: {target_personalities}")
    except Exception as e:
        logger.critical(f"Failed to load dataset '{args.dataset_name}': {e}", exc_info=True)
        return

    if not args.skip_training:
        train_adapters(args, target_personalities)
    else:
        logger.info("Skipping training phase as requested.")

    df_results = run_evaluation(args, target_personalities)
    
    if not df_results.empty:
        analyze_and_report(df_results, target_personalities, args)

    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info(f"EXPERIMENT COMPLETE. Total runtime: {total_time / 60:.2f} minutes.")
    logger.info("="*80)


if __name__ == "__main__":
    main()