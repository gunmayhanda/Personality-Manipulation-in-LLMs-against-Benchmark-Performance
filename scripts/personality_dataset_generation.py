"""
This script generates a contrastive dataset for personality steering by creating
"antithetical" responses for an existing personality-labeled dataset.

The process is as follows:
1.  Loads the official 'train' and 'test' splits from the
    'holistic-ai/personality_manipulation' dataset.
2.  For each split, it identifies the unique prompts (question + topic).
3.  For each unique prompt, it uses an Azure OpenAI model to generate a response
    embodying the opposite personality trait (e.g., generates an 'introversion'
    response for a prompt that originally had an 'extraversion' answer).
4.  It merges the original high-trait responses with the newly generated
    low-trait responses to create contrastive pairs.
5.  Finally, it saves the completed contrastive training and test sets to
    separate CSV files.
"""
import os
import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from openai import AzureOpenAI
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login


# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Trait Definitions & Mappings ---
COMPREHENSIVE_TRAIT_DESCRIPTIONS = {
    "openness": "You are intellectually curious, creative, and imaginative...",
    "closedness to experience": "You are pragmatic, conventional, and data-driven...",
    "conscientiousness": "You are organized, disciplined, and dependable...",
    "low conscientiousness": "You are spontaneous, flexible, and sometimes disorganized...",
    "extraversion": "You are energetic, assertive, and sociable...",
    "introversion": "You are reserved, deliberate, and independent...",
    "agreeableness": "You are cooperative, compassionate, and considerate...",
    "disagreeableness": "You are competitive, critical, and skeptical...",
    "neuroticism": "You tend to experience negative emotions like anxiety, anger, and sadness...",
    "emotional stability": "You are calm, resilient, and secure...",
}
TRAIT_OPPOSITES = {
    "extraversion": "introversion",
    "agreeableness": "disagreeableness",
    "conscientiousness": "low conscientiousness",
    "neuroticism": "emotional stability",
    "openness": "closedness to experience"
}
HIGH_TRAITS_FROM_LOW = {v: k for k, v in TRAIT_OPPOSITES.items()}


class GPTAgent:
    """A wrapper for the Azure OpenAI API to generate personality-driven text."""

    def __init__(self, client: AzureOpenAI, deployment_name: str):
        self.client = client
        self.deployment_name = deployment_name
        self.prompt_template = (
            "Instruction: You are an AI assistant. Your task is to answer the user's question from a specific personality perspective, "
            "which is described in the 'Target Personality Profile'. Your response should be a single, concise paragraph that reflects "
            "this personality when discussing the 'Edit Topic'.\n\n"
            "--- Your Task ---\n"
            "Target Personality Profile: {profile}\n"
            "Edit Topic: {topic}\n"
            "Question: {question}\n"
            "Answer:"
        )

    def invoke(self, profile: str, topic: str, question: str) -> tuple[str, str]:
        """Generates a response based on a personality profile and prompt."""
        prompt = self.prompt_template.format(profile=profile, topic=topic, question=question)
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            choice = response.choices[0]
            content = choice.message.content.strip() if choice.message.content else ""
            return content, choice.finish_reason
        except Exception as e:
            logger.error(f"Azure API call failed: {e}")
            return f"ERROR: API_CALL_FAILED - {str(e)}", "exception"


class ContrastiveDataGenerator:
    """Orchestrates the creation of the contrastive personality dataset."""

    def __init__(self, agent: GPTAgent, args: argparse.Namespace):
        self.agent = agent
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_and_prepare_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads and standardizes the official train and test splits."""
        logger.info(f"Loading '{self.args.dataset_name}' from Hugging Face Hub...")
        try:
            train_ds = load_dataset(self.args.dataset_name, split="train")
            test_ds = load_dataset(self.args.dataset_name, split="test")
            
            df_train = train_ds.to_pandas()
            df_test = test_ds.to_pandas()

            df_train = self._standardize_columns(df_train)
            df_test = self._standardize_columns(df_test)
            
            logger.info(f"Loaded train split ({len(df_train)} rows) and test split ({len(df_test)} rows).")
            return df_train, df_test
        except Exception as e:
            logger.critical(f"Failed to load or process dataset splits: {e}", exc_info=True)
            sys.exit(1)

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names to lowercase with underscores."""
        df.columns = [col.replace(' ', '_').lower() for col in df.columns]
        required = ['question', 'edit_topic', 'answer', 'target_personality']
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame is missing required columns. Found: {df.columns.tolist()}")
        return df

    def create_contrastive_set(self, original_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """Generates and merges antithetical data for a given data split."""
        logger.info(f"Processing '{split_name}' split...")
        cache_path = self.output_dir / f"{split_name}_antithetical_cache.csv"

        if cache_path.exists():
            logger.info(f"Found cache at '{cache_path}'. Loading generated responses.")
            generated_df = pd.read_csv(cache_path)
        else:
            logger.info("No cache found. Generating antithetical responses...")
            unique_prompts_df = original_df[['question', 'edit_topic']].drop_duplicates().reset_index(drop=True)
            logger.info(f"Found {len(unique_prompts_df)} unique prompts in the '{split_name}' split.")
            generated_df = self._generate_and_cache_responses(unique_prompts_df, cache_path)

        logger.info(f"Merging original '{split_name}' data with generated responses.")
        contrastive_df = self._merge_data(original_df, generated_df)
        return contrastive_df

    def _generate_and_cache_responses(self, prompts_df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
        """Generates antithetical responses and saves them to a cache file."""
        generated_responses = []
        total_calls = len(prompts_df) * len(TRAIT_OPPOSITES)
        
        with tqdm(total=total_calls, desc=f"Generating for {cache_path.stem}") as pbar:
            for _, row in prompts_df.iterrows():
                for low_trait, high_trait in HIGH_TRAITS_FROM_LOW.items():
                    profile = COMPREHENSIVE_TRAIT_DESCRIPTIONS[low_trait]
                    response, reason = self.agent.invoke(profile, row['edit_topic'], row['question'])
                    
                    if not response and reason != 'stop':
                        logger.warning(f"Generation issue for topic '{row['edit_topic']}' with trait '{low_trait}'. Reason: {reason}")

                    generated_responses.append({
                        'question': row['question'],
                        'edit_topic': row['edit_topic'],
                        'low_trait_label': low_trait,
                        'low_trait_response': response,
                        'trait_dimension': high_trait  # Add dimension for merging
                    })
                    pbar.update(1)
                    time.sleep(self.args.api_delay)
        
        generated_df = pd.DataFrame(generated_responses)
        generated_df.to_csv(cache_path, index=False)
        logger.info(f"Saved generated responses to cache: '{cache_path}'")
        return generated_df

    @staticmethod
    def _merge_data(original_df: pd.DataFrame, generated_df: pd.DataFrame) -> pd.DataFrame:
        """Merges original high-trait data with generated low-trait data."""
        # Prepare original dataframe
        df_orig = original_df.copy()
        df_orig.rename(columns={'answer': 'high_trait_response', 'target_personality': 'high_trait_label'}, inplace=True)
        df_orig['trait_dimension'] = df_orig['high_trait_label']
        
        # Merge
        final_df = pd.merge(
            df_orig[['question', 'edit_topic', 'trait_dimension', 'high_trait_label', 'high_trait_response']],
            generated_df[['question', 'edit_topic', 'trait_dimension', 'low_trait_label', 'low_trait_response']],
            on=['question', 'edit_topic', 'trait_dimension'],
            how='inner'
        )
        
        if final_df.empty:
            raise ValueError("Merged dataset is empty. Check for inconsistencies in 'question', 'edit_topic', or 'trait_dimension' columns.")
        
        logger.info(f"Successfully merged data, resulting in {len(final_df)} contrastive pairs.")
        return final_df


def setup_clients() -> tuple[AzureOpenAI | None, str | None]:
    """Initializes and validates the Azure OpenAI client from environment variables."""
    try:
        load_dotenv()
        logger.info("Loaded environment variables from .env file.")

        if hf_token := os.getenv("HF_TOKEN"):
            login(token=hf_token)
            logger.info("Successfully logged into Hugging Face Hub.")

        api_key = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if not all([api_key, endpoint, version, deployment]):
            raise ValueError("One or more required Azure OpenAI environment variables are missing.")

        client = AzureOpenAI(api_key=api_key, api_version=version, azure_endpoint=endpoint)
        logger.info("Azure OpenAI client initialized successfully.")
        return client, deployment
    except Exception as e:
        logger.critical(f"Client initialization failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate a contrastive personality dataset.")
    parser.add_argument("--dataset_name", type=str, default="holistic-ai/personality_manipulation", help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files.")
    parser.add_argument("--api_delay", type=float, default=0.5, help="Delay in seconds between API calls to prevent rate limiting.")
    args = parser.parse_args()

    azure_client, deployment_name = setup_clients()
    agent = GPTAgent(azure_client, deployment_name)
    generator = ContrastiveDataGenerator(agent, args)
    
    # 1. Load the official splits
    df_train_orig, df_test_orig = generator.load_and_prepare_splits()
    
    # 2. Process the training set
    contrastive_train_df = generator.create_contrastive_set(df_train_orig, "train")
    
    # 3. Process the test set
    contrastive_test_df = generator.create_contrastive_set(df_test_orig, "test")

    # 4. Save final files
    train_path = Path(args.output_dir) / "personality_contrastive_data_train.csv"
    test_path = Path(args.output_dir) / "personality_contrastive_data_test.csv"
    
    contrastive_train_df.to_csv(train_path, index=False)
    logger.info(f"Successfully saved final training set to '{train_path}'")
    contrastive_test_df.to_csv(test_path, index=False)
    logger.info(f"Successfully saved final test set to '{test_path}'")

    logger.info("\n=== Process completed successfully! ===")
    logger.info("Final dataset statistics:")
    logger.info(f"  - Total contrastive TRAIN pairs: {len(contrastive_train_df)}")
    logger.info(f"  - Total contrastive TEST pairs:  {len(contrastive_test_df)}")
    logger.info(f"  - Trait dimensions covered:      {contrastive_train_df['trait_dimension'].nunique()}")


if __name__ == "__main__":
    main()