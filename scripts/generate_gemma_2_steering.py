"""
A consolidated pipeline for extracting, calibrating, and validating personality-based
steering vectors for large language models.

This script encapsulates a multi-stage workflow:
1.  EXTRACT: Generates steering vectors for the Big Five personality traits by
    contrasting model activations on high- and low-trait text pairs.
2.  PURIFY: (Optional but recommended) Orthogonalizes the 'openness' vector against
    its primary contaminants ('conscientiousness', 'extraversion') to improve its
    specificity.
3.  CALIBRATE: Employs an efficient "Intelligent Binary Scout" algorithm to find the
    optimal steering layer and strength for each personality vector.
4.  VALIDATE: Performs a rigorous, multi-run statistical validation of the optimal
    parameters, comparing steered model performance against a baseline and calculating
    confidence intervals and p-values.
5.  DEMO: Provides a qualitative demonstration of the final steering vectors on a
    set of diverse prompts.

Usage:
    # Run the full pipeline from start to finish
    python run_steering_experiment.py --all

    # Run only a specific stage
    python run_steering_experiment.py --extract
    python run_steering_experiment.py --calibrate

    # Run validation with more samples and runs
    python run_steering_experiment.py --validate --n_samples_valid 200 --n_runs 5
"""
import os
import argparse
import logging
import time
import json
import textwrap
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from statsmodels.stats.contingency_tables import mcnemar


# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# These can be overridden by command-line arguments
DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"
DEFAULT_OUTPUT_DIR = "persona_steering_results"
DEFAULT_TRAIN_CSV = "path/to/your/personality_contrastive_data_train.csv"
DEFAULT_TEST_CSV = "path/to/your/personality_contrastive_data_test.csv"

# --- Experiment Constants ---
PERSONALITY_TRAITS = ["extraversion", "agreeableness", "neuroticism", "openness", "conscientiousness"]
TARGET_LAYERS = [5, 10, 15, 20]
GENERATION_MAX_TOKENS = 100
CLASSIFIER_PERSONALITY = "holistic-ai/personality_classifier"
CLASSIFIER_GIBBERISH = "madhurjindal/autonlp-Gibberish-Detector-492513457"


# ==============================================================================
# Persona Steering Pipeline Class
# ==============================================================================

class PersonaSteeringPipeline:
    """Manages the entire workflow for persona steering vector experiments."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(self.args.output_dir)
        self.vectors_cache_dir = self.output_dir / "vectors_cache"
        self.calibration_results_path = self.output_dir / "optimal_params.json"
        
        self.output_dir.mkdir(exist_ok=True)
        self.vectors_cache_dir.mkdir(exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.p_classifier = None
        self.g_classifier = None
        self.df_train = None
        self.df_test = None

    def _load_assets(self):
        """Loads all necessary models and data, avoiding redundant loads."""
        if self.model:
            return

        logger.info("--- Loading all necessary assets (this may take a moment) ---")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        device = self.model.device
        self.p_classifier = pipeline("text-classification", model=CLASSIFIER_PERSONALITY, device=device)
        self.g_classifier = pipeline("text-classification", model=CLASSIFIER_GIBBERISH, device=device)

        try:
            self.df_train = pd.read_csv(self.args.train_csv)
            self.df_test = pd.read_csv(self.args.test_csv)
        except FileNotFoundError as e:
            logger.error(f"FATAL: Could not find data CSV at {e.filename}. Please check paths.")
            raise
        logger.info("✓ All assets loaded successfully.")

    def _get_vector_path(self, trait: str, version: str = "") -> Path:
        """Constructs a standardized path for a vector file."""
        model_id = self.args.model_name.split("/")[-1]
        filename = f"{trait}{version}_{model_id}.pt"
        return self.vectors_cache_dir / filename

    # --- STAGE 1: EXTRACTION ---
    def extract_vectors(self):
        """Extracts and saves steering vectors for each personality trait."""
        self._load_assets()
        logger.info("\n" + "="*80)
        logger.info(" " * 28 + "STAGE 1: EXTRACT VECTORS")
        logger.info("="*80)

        for trait in self.args.traits:
            vector_path = self._get_vector_path(trait)
            if vector_path.exists() and not self.args.force_rerun:
                logger.info(f"Vectors for '{trait}' already exist. Skipping.")
                continue

            logger.info(f"--- Extracting vectors for trait: {trait.upper()} ---")
            df_trait = self.df_train[self.df_train['trait_dimension'] == trait].copy()
            if df_trait.empty:
                logger.warning(f"No data for trait '{trait}' in training set. Skipping.")
                continue
            
            persona_vectors = {}
            for layer_idx in TARGET_LAYERS:
                activations = self._get_activations_for_layer(df_trait, layer_idx)
                diff_vector = (activations['pos'] - activations['neg']).mean(dim=0)
                norm_vector = (diff_vector / torch.norm(diff_vector)).to(torch.bfloat16)
                persona_vectors[layer_idx] = norm_vector
            
            torch.save(persona_vectors, vector_path)
            logger.info(f"✓ Vectors for '{trait}' saved to {vector_path}")

    def _get_activations_for_layer(self, df: pd.DataFrame, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Helper to get positive and negative activations for a given layer."""
        captured_activations = {}
        def hook_fn(module, input, output): captured_activations['current'] = output.cpu()
        target_module = self.model.model.layers[layer_idx].post_attention_layernorm
        handle = target_module.register_forward_hook(hook_fn)

        pos_activations, neg_activations = [], []
        batch_size = 8
        
        for i in tqdm(range(0, len(df), batch_size), desc=f"  Layer {layer_idx} Batches"):
            batch_df = df.iloc[i:i+batch_size]
            
            # Positive activations
            pos_chats = [[{"role": "user", "content": r['question']}, {"role": "model", "content": r['high_trait_response']}] for _, r in batch_df.iterrows()]
            pos_texts = [self.tokenizer.apply_chat_template(c, tokenize=False) for c in pos_chats]
            inputs = self.tokenizer(pos_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            self.model(**inputs)
            pos_activations.append(captured_activations['current'][:, -1, :].clone())
            
            # Negative activations
            neg_chats = [[{"role": "user", "content": r['question']}, {"role": "model", "content": r['low_trait_response']}] for _, r in batch_df.iterrows()]
            neg_texts = [self.tokenizer.apply_chat_template(c, tokenize=False) for c in neg_chats]
            inputs = self.tokenizer(neg_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            self.model(**inputs)
            neg_activations.append(captured_activations['current'][:, -1, :].clone())
        
        handle.remove()
        torch.cuda.empty_cache()
        return {'pos': torch.cat(pos_activations, dim=0), 'neg': torch.cat(neg_activations, dim=0)}

    # --- STAGE 2: PURIFICATION ---
    def purify_vectors(self):
        """Purifies the 'openness' vector using orthogonal projection."""
        logger.info("\n" + "="*80)
        logger.info(" " * 26 + "STAGE 2: PURIFY 'OPENNESS' VECTOR")
        logger.info("="*80)

        trait_to_purify = 'openness'
        contaminant_traits = ['conscientiousness', 'extraversion']
        purified_path = self._get_vector_path(trait_to_purify, version="_purified")

        if purified_path.exists() and not self.args.force_rerun:
            logger.info(f"Purified vector for '{trait_to_purify}' already exists. Skipping.")
            return

        try:
            v_target_dict = torch.load(self._get_vector_path(trait_to_purify))
            contaminant_dicts = [torch.load(self._get_vector_path(c)) for c in contaminant_traits]
        except FileNotFoundError as e:
            logger.error(f"Cannot purify: Missing required vector file: {e.filename}. Please run --extract first.")
            return

        purified_vectors = {}
        for layer in TARGET_LAYERS:
            v_target = v_target_dict[layer].to(dtype=torch.float32)
            contaminant_matrix = torch.stack([d[layer].to(dtype=torch.float32) for d in contaminant_dicts], dim=1)
            
            Q, _ = torch.linalg.qr(contaminant_matrix)
            projection = torch.zeros_like(v_target)
            for i in range(Q.shape[1]):
                basis_vector = Q[:, i]
                projection += torch.dot(v_target, basis_vector) * basis_vector
            
            v_purified = v_target - projection
            purified_vectors[layer] = (v_purified / torch.norm(v_purified)).to(dtype=torch.bfloat16)
        
        torch.save(purified_vectors, purified_path)
        logger.info(f"✓ Purified '{trait_to_purify}' vector saved to {purified_path}")

    # --- STAGE 3: CALIBRATION ---
    def calibrate_vectors(self):
        """Finds optimal layer and strength for each vector using an intelligent search."""
        self._load_assets()
        logger.info("\n" + "="*80)
        logger.info(" " * 29 + "STAGE 3: CALIBRATE VECTORS")
        logger.info("="*80)

        all_results = []
        for trait in self.args.traits:
            version = "_purified" if trait == 'openness' else ""
            vector_path = self._get_vector_path(trait, version)
            if not vector_path.exists():
                logger.warning(f"Vector for '{trait}' (version: '{version or 'base'}') not found. Skipping calibration.")
                continue
            vectors = torch.load(vector_path)

            for layer in TARGET_LAYERS:
                strength, acc, clean, score = self._calibrate_single_vector(trait, layer, vectors)
                all_results.append({
                    'trait': trait, 'layer': layer, 'strength': strength,
                    'accuracy': acc, 'cleanliness': clean, 'score': score
                })
        
        if not all_results:
            logger.error("Calibration failed: no results were generated.")
            return

        df = pd.DataFrame(all_results)
        best_idx = df.groupby('trait')['score'].idxmax()
        optimal_params_df = df.loc[best_idx]
        
        logger.info("\n--- Optimal Parameters Found ---")
        print(optimal_params_df.to_string(index=False))

        # Save results to JSON for the validation stage
        optimal_params_dict = optimal_params_df.to_dict('records')
        with open(self.calibration_results_path, 'w') as f:
            json.dump(optimal_params_dict, f, indent=4)
        logger.info(f"✓ Optimal parameters saved to {self.calibration_results_path}")

    def _calibrate_single_vector(self, trait: str, layer: int, vectors: dict):
        """Performs the 'Intelligent Binary Scout' search for one vector."""
        logger.info(f"--- Calibrating '{trait.upper()}' at Layer {layer} ---")
        prompts = self.df_test[self.df_test['trait_dimension'] == trait].drop_duplicates('question').head(self.args.n_samples_calib)['question'].tolist()
        
        search_history = []
        def evaluate(strength):
            acc, clean = self._check_strength(prompts, trait, vectors, layer, strength)
            search_history.append({'strength': strength, 'accuracy': acc, 'cleanliness': clean})
            return acc, clean

        lo, hi, best_safe_strength, max_safe_score = -300.0, 300.0, 0.0, -1.0
        
        # Phase 1: Binary Scout
        for _ in range(8):
            p1, p2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            acc1, clean1 = evaluate(p1)
            score1 = acc1 * clean1 if clean1 >= 0.9 else -1
            acc2, clean2 = evaluate(p2)
            score2 = acc2 * clean2 if clean2 >= 0.9 else -1
            
            if score1 > max_safe_score: max_safe_score, best_safe_strength = score1, p1
            if score2 > max_safe_score: max_safe_score, best_safe_strength = score2, p2
            if score1 > score2: hi = p2
            else: lo = p1
            
        # Phase 2: Local Zoom
        final_strength, final_max_score = best_safe_strength, max_safe_score
        zoom_range = np.linspace(best_safe_strength - 30.0, best_safe_strength + 30.0, 11)
        for s in tqdm(zoom_range, desc=f"Zooming L{layer}", leave=False):
            acc, clean = evaluate(s)
            score = acc * clean if clean >= 0.9 else -1
            if score > final_max_score:
                final_max_score, final_strength = score, s
        
        final_acc, final_clean = self._check_strength(prompts, trait, vectors, layer, final_strength)
        
        if self.args.plot:
            history_df = pd.DataFrame(search_history).drop_duplicates('strength').sort_values('strength')
            self._plot_search_history(history_df, trait, layer, final_strength)

        return final_strength, final_acc, final_clean, final_max_score

    def _check_strength(self, prompts, trait, vectors, layer, strength):
        """Helper to test a single strength and return metrics."""
        hits, clean_count = 0, 0
        for prompt in prompts:
            response = self._generate_steered(prompt, vectors, layer, strength)
            if not response: continue
            
            p_pred = self.p_classifier(response, truncation=True)[0]['label']
            g_pred = self.g_classifier(response, truncation=True)[0]['label']
            
            if p_pred == trait: hits += 1
            if g_pred == 'clean': clean_count += 1
            
        return hits / len(prompts), clean_count / len(prompts)
    
    def _plot_search_history(self, history_df, trait, layer, final_strength):
        """Visualizes the points tested during the search process."""
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(history_df['strength'], history_df['accuracy'], 'o', color='tab:blue', label='Accuracy')
        ax1.set_xlabel('Steering Strength'); ax1.set_ylabel('Accuracy', color='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(history_df['strength'], history_df['cleanliness'], 'x', color='tab:green', label='Cleanliness')
        ax2.set_ylabel('Cleanliness', color='tab:green')
        ax1.axvline(x=final_strength, color='r', linestyle=':', lw=2.5, label=f'Final Strength ({final_strength:.2f})')
        fig.suptitle(f"Search History for '{trait.title()}' at Layer {layer}")
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9)); plt.grid(True, alpha=0.3)
        plot_path = self.output_dir / f"calibration_plot_{trait}_L{layer}.png"
        plt.savefig(plot_path); plt.close()
        logger.info(f"Calibration plot saved to {plot_path}")

    # --- STAGE 4: VALIDATION ---
    def validate_vectors(self):
        """Performs multi-run statistical validation of the optimal parameters."""
        self._load_assets()
        logger.info("\n" + "="*80)
        logger.info(" " * 28 + "STAGE 4: VALIDATE VECTORS")
        logger.info("="*80)

        if not self.calibration_results_path.exists():
            logger.error(f"Cannot run validation: optimal parameters file not found at {self.calibration_results_path}. Please run --calibrate first.")
            return
        with open(self.calibration_results_path, 'r') as f:
            optimal_params = json.load(f)

        final_summary = []
        for config in optimal_params:
            trait, layer, strength = config['trait'], int(config['layer']), config['strength']
            version = "_purified" if trait == 'openness' else ""
            vector_path = self._get_vector_path(trait, version)
            if not vector_path.exists(): continue
            vectors = torch.load(vector_path)

            logger.info(f"--- Validating '{trait.upper()}' (L{layer}, S={strength:.1f}) ---")
            prompts = self.df_test[self.df_test['trait_dimension'] == trait].drop_duplicates('question').head(self.args.n_samples_valid)['question'].tolist()
            
            run_metrics = {'baseline': [], 'steered': [], 'clean': []}
            paired_outcomes = []

            for run in range(self.args.n_runs):
                logger.info(f"  Starting generation run {run + 1}/{self.args.n_runs}")
                base_hits, steered_hits, clean_hits = 0, 0, 0
                for prompt in tqdm(prompts, desc=f"Run {run+1}", leave=False):
                    base_resp = self._generate_steered(prompt, vectors, layer, 0.0)
                    steered_resp = self._generate_steered(prompt, vectors, layer, strength)
                    
                    base_correct = self.p_classifier(base_resp, truncation=True)[0]['label'] == trait if base_resp else False
                    if base_correct: base_hits += 1
                    
                    if steered_resp:
                        steered_correct = self.p_classifier(steered_resp, truncation=True)[0]['label'] == trait
                        is_clean = self.g_classifier(steered_resp, truncation=True)[0]['label'] == 'clean'
                        if steered_correct: steered_hits += 1
                        if is_clean: clean_hits += 1
                        if run == 0: paired_outcomes.append((base_correct, steered_correct))

                run_metrics['baseline'].append(base_hits / len(prompts))
                run_metrics['steered'].append(steered_hits / len(prompts))
                run_metrics['clean'].append(clean_hits / len(prompts))

            # Calculate stats
            contingency = pd.crosstab(pd.Series([p[0] for p in paired_outcomes]), pd.Series([p[1] for p in paired_outcomes])).reindex(index=[False, True], columns=[False, True], fill_value=0)
            p_value = mcnemar(contingency, exact=True).pvalue
            
            def get_ci(data):
                mean = np.mean(data)
                if len(data) < 2: return f"{mean:.1%}"
                std_err = np.std(data, ddof=1) / np.sqrt(len(data))
                return f"{mean:.1%} (±{1.96 * std_err:.1%})"
                
            final_summary.append({
                'Trait': trait.title(), 'Layer': layer, 'Strength': f"{strength:.1f}",
                'Baseline Acc (95% CI)': get_ci(run_metrics['baseline']),
                'Steered Acc (95% CI)': get_ci(run_metrics['steered']),
                'Cleanliness (95% CI)': get_ci(run_metrics['clean']),
                'P-Value': f"{p_value:.4f}",
                'Significant': p_value < 0.05
            })
        
        logger.info("\n" + "="*100)
        logger.info(f" " * 30 + f"DEFINITIVE STATISTICAL VALIDATION REPORT")
        logger.info("="*100)
        report_df = pd.DataFrame(final_summary)
        print(report_df.to_string(index=False))

    def _generate_steered(self, prompt, vectors, layer, strength):
        """Helper for a single steered generation."""
        hook_handle = None
        try:
            if strength != 0.0:
                vec = vectors[layer].to(self.model.device)
                def hook(module, input, output): return output + vec.to(output.dtype) * strength
                target_module = self.model.model.layers[layer].post_attention_layernorm
                hook_handle = target_module.register_forward_hook(hook)
            
            chat = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                input_ids=inputs, max_new_tokens=GENERATION_MAX_TOKENS, do_sample=True,
                temperature=0.7, pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0, inputs.shape[1]:], skip_special_tokens=True).strip()
        finally:
            if hook_handle: hook_handle.remove()

    # --- STAGE 5: DEMO ---
    def run_demo(self):
        """Runs a qualitative demo of the final steering vectors."""
        self._load_assets()
        logger.info("\n" + "="*80)
        logger.info(" " * 30 + "STAGE 5: QUALITATIVE DEMO")
        logger.info("="*80)
        
        if not self.calibration_results_path.exists():
            logger.error("Cannot run demo: optimal parameters file not found. Please run --calibrate first.")
            return
        with open(self.calibration_results_path, 'r') as f:
            optimal_params = json.load(f)

        test_prompts = [
            "I have a completely free weekend with no obligations. Describe the ideal plan for Saturday.",
            "My boss just gave me some unexpected critical feedback on a project I worked hard on. How should I react?",
            "I've been assigned to a group project, but my teammates seem unmotivated. What should I do?"
        ]

        for config in optimal_params:
            trait, layer, strength = config['trait'], int(config['layer']), config['strength']
            version = "_purified" if trait == 'openness' else ""
            vector_path = self._get_vector_path(trait, version)
            if not vector_path.exists(): continue
            vectors = torch.load(vector_path)

            print("\n\n" + "#"*80)
            print(f"#" + " "*25 + f"VERIFYING: {trait.upper()} (L{layer}, S={strength:.1f})")
            print("#"*80)

            for prompt in test_prompts:
                print("\n" + "-"*80)
                print(f"PROMPT: \"{prompt}\"")
                print("-"*80)
                
                base_resp = self._generate_steered(prompt, vectors, layer, 0.0)
                pos_resp = self._generate_steered(prompt, vectors, layer, strength)
                neg_resp = self._generate_steered(prompt, vectors, layer, -strength)
                
                print("\n--- BASELINE (Strength 0.0) ---")
                print(textwrap.fill(base_resp, width=80))
                print(f"\n--- POSITIVE STEER (+{strength:.1f}) | High {trait.title()} ---")
                print(textwrap.fill(pos_resp, width=80))
                print(f"\n--- NEGATIVE STEER (-{strength:.1f}) | Low {trait.title()} ---")
                print(textwrap.fill(neg_resp, width=80))


def main():
    parser = argparse.ArgumentParser(description="A consolidated pipeline for persona steering vector experiments.")
    
    # --- Execution Control ---
    parser.add_argument("--all", action="store_true", help="Run all stages of the pipeline in order.")
    parser.add_argument("--extract", action="store_true", help="Run only Stage 1: Vector Extraction.")
    parser.add_argument("--purify", action="store_true", help="Run only Stage 2: Purify 'openness' vector.")
    parser.add_argument("--calibrate", action="store_true", help="Run only Stage 3: Vector Calibration.")
    parser.add_argument("--validate", action="store_true", help="Run only Stage 4: Statistical Validation.")
    parser.add_argument("--demo", action="store_true", help="Run only Stage 5: Qualitative Demo.")
    parser.add_argument("--force_rerun", action="store_true", help="Force re-running a stage even if output files exist.")

    # --- Configuration ---
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Name of the Hugging Face model to use.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save all outputs.")
    parser.add_argument("--train_csv", type=str, default=DEFAULT_TRAIN_CSV, help="Path to the training data CSV.")
    parser.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV, help="Path to the test data CSV.")
    parser.add_argument("--traits", nargs='+', default=PERSONALITY_TRAITS, help="List of traits to process.")

    # --- Hyperparameters ---
    parser.add_argument("--n_samples_calib", type=int, default=15, help="Number of samples per check during calibration.")
    parser.add_argument("--n_samples_valid", type=int, default=100, help="Number of samples for the final validation.")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of full generation runs for statistical validation.")
    parser.add_argument("--plot", action="store_true", help="Generate and save calibration plots.")
    
    args = parser.parse_args()
    
    # --- Environment Setup ---
    load_dotenv()
    if hf_token := os.getenv("HF_TOKEN"):
        login(token=hf_token)

    start_time = time.time()
    pipeline = PersonaSteeringPipeline(args)
    
    # --- Pipeline Execution ---
    run_all = args.all or not any([args.extract, args.purify, args.calibrate, args.validate, args.demo])
    
    if run_all or args.extract:
        pipeline.extract_vectors()
    if run_all or args.purify:
        pipeline.purify_vectors()
    if run_all or args.calibrate:
        pipeline.calibrate_vectors()
    if run_all or args.validate:
        pipeline.validate_vectors()
    if run_all or args.demo:
        pipeline.run_demo()

    total_time = time.time() - start_time
    logger.info(f"\n--- Experiment Complete. Total runtime: {total_time / 60:.2f} minutes. ---")

if __name__ == "__main__":
    main()