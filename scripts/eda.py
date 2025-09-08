"""
A comprehensive tool for Exploratory Data Analysis (EDA) and quality assessment of
a contrastive personality dataset.

This script performs several phases of analysis, which can be run individually
or all at once:
1.  FOUNDATION: Loads and merges multiple data sources, providing basic stats.
2.  LINGUISTIC: Analyzes lexical signatures (Word Clouds, TF-IDF), response
    patterns (length, complexity), and sentiment.
3.  PURITY & CONTAMINATION: Assesses cross-trait vocabulary overlap and uses a
    classifier to filter for high-purity data subsets.
4.  BIAS & HEURISTICS: Screens for sensitive keywords and identifies highly
    contrastive pairs for manual review.

Usage:
    # Run all analysis stages on the specified input files
    python analyze_contrastive_dataset.py --all \
        --input_files path/to/train.csv path/to/test.csv \
        --output_dir ./analysis_results

    # Run only the linguistic analysis and data purity stages
    python analyze_contrastive_dataset.py --linguistic --purity \
        --input_files path/to/train.csv \
        --purity_threshold 0.6
"""
import os
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

# --- Constants ---
CLASSIFIER_MODEL_NAME = "holistic-ai/personality_classifier"
OPPOSITE_TRAIT_MAP = {
    'extraversion': 'Introversion', 'agreeableness': 'Disagreeableness',
    'neuroticism': 'Emotional Stability', 'openness': 'Closedness to Experience',
    'conscientiousness': 'Spontaneity'
}

class DatasetAnalyzer:
    """Encapsulates the full EDA and analysis workflow for the dataset."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.df: pd.DataFrame | None = None
        self.classifier: Any = None
        self.trait_vocab: Dict[str, set] = {}

    def load_and_prepare_data(self):
        """Loads and merges data from multiple source files."""
        if self.df is not None:
            return
        
        logger.info("[PHASE 1] Loading and preparing dataset foundation...")
        all_dfs = []
        for path_str in self.args.input_files:
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"File not found, skipping: {path}")
                continue
            try:
                temp_df = pd.read_csv(path)
                temp_df['source_file'] = path.stem
                all_dfs.append(temp_df)
                logger.info(f"  - Loaded {len(temp_df)} rows from '{path.name}'")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")

        if not all_dfs:
            logger.critical("No data files were successfully loaded. Exiting.")
            sys.exit(1)

        self.df = pd.concat(all_dfs, ignore_index=True).dropna(
            subset=['high_trait_response', 'low_trait_response']
        )
        logger.info(f"Successfully combined datasets. Total valid rows: {len(self.df)}")
        logger.info("Data Sources Breakdown:\n" + self.df['source_file'].value_counts().to_string())
        logger.info("Overall Trait Balance:\n" + self.df['trait_dimension'].value_counts().to_string())

    def _load_classifier(self):
        """Loads the personality classifier model, only when needed."""
        if self.classifier:
            return
        
        logger.info(f"Loading classifier: {CLASSIFIER_MODEL_NAME}...")
        try:
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_str)
            logger.info(f"Using device: {device_str}")
            
            model = AutoModelForSequenceClassification.from_pretrained(
                CLASSIFIER_MODEL_NAME, torch_dtype=torch.bfloat16
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)
            
            self.classifier = pipeline(
                "text-classification", model=model, tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1, top_k=None
            )
            logger.info("✓ Personality classifier loaded successfully.")
        except Exception as e:
            logger.error(f"Could not load personality classifier: {e}", exc_info=True)
            raise

    def analyze_linguistic_features(self):
        """Runs all linguistic analyses (word clouds, TF-IDF, patterns, sentiment)."""
        if self.df is None: self.load_and_prepare_data()
        
        logger.info("\n[PHASE 2] Starting comprehensive linguistic analysis...")
        self._analyze_word_clouds()
        self._analyze_tfidf()
        self._analyze_response_patterns()
        self._analyze_sentiment()
        
    def _analyze_word_clouds(self):
        logger.info("Generating trait-specific word clouds...")
        for trait in self.df['trait_dimension'].unique():
            # High trait
            text_high = ' '.join(self.df[self.df['trait_dimension'] == trait]['high_trait_response'])
            self._create_and_save_wordcloud(text_high, f'High-{trait.capitalize()}', 'viridis', 'white')
            # Low trait
            opposite_name = OPPOSITE_TRAIT_MAP.get(trait, f"Low-{trait.capitalize()}")
            text_low = ' '.join(self.df[self.df['trait_dimension'] == trait]['low_trait_response'])
            self._create_and_save_wordcloud(text_low, opposite_name, 'plasma', 'black')

    def _create_and_save_wordcloud(self, text, title, colormap, bgcolor):
        path = self.output_dir / f"wordcloud_{title.replace(' ', '_')}.png"
        wordcloud = WordCloud(width=800, height=400, background_color=bgcolor, colormap=colormap, collocations=False).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear'); plt.axis('off'); plt.title(title, fontsize=16)
        plt.savefig(path, bbox_inches='tight'); plt.close()
        logger.info(f"  - Saved word cloud to '{path}'")

    def _analyze_tfidf(self):
        logger.info("Calculating quantitative lexical signatures (TF-IDF)...")
        docs, labels = [], []
        for trait in self.df['trait_dimension'].unique():
            labels.append(f"High-{trait.capitalize()}"); docs.append(' '.join(self.df[self.df['trait_dimension'] == trait]['high_trait_response']))
            labels.append(OPPOSITE_TRAIT_MAP.get(trait, f"Low-{trait.capitalize()}")); docs.append(' '.join(self.df[self.df['trait_dimension'] == trait]['low_trait_response']))
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        
        for i, label in enumerate(labels):
            feature_idx = tfidf_matrix[i,:].nonzero()[1]
            scores = zip(feature_idx, [tfidf_matrix[i, x] for x in feature_idx])
            top_words = [feature_names[i] for i, s in sorted(scores, key=lambda x: x[1], reverse=True)[:10]]
            self.trait_vocab[label] = set(top_words)
            logger.info(f"  - Top 10 keywords for {label}: {', '.join(top_words)}")

    def _analyze_response_patterns(self):
        logger.info("Analyzing response patterns (length and complexity)...")
        self.df['high_len'] = self.df['high_trait_response'].str.split().str.len()
        self.df['low_len'] = self.df['low_trait_response'].str.split().str.len()
        self.df['high_comp'] = self.df['high_trait_response'].apply(textstat.flesch_kincaid_grade)
        self.df['low_comp'] = self.df['low_trait_response'].apply(textstat.flesch_kincaid_grade)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        sns.boxplot(data=self.df[['high_len', 'low_len']], ax=axes[0], palette=['skyblue', 'salmon'], notch=True)
        axes[0].set_xticklabels(['High-Trait Length', 'Low-Trait Length']); axes[0].set_title('Distribution of Response Length (Words)')
        sns.boxplot(data=self.df[['high_comp', 'low_comp']], ax=axes[1], palette=['skyblue', 'salmon'], notch=True)
        axes[1].set_xticklabels(['High-Trait Complexity', 'Low-Trait Complexity']); axes[1].set_title('Distribution of Response Complexity (Grade Level)')
        
        path = self.output_dir / "response_patterns_plot.png"
        plt.savefig(path); plt.close()
        logger.info(f"  - Saved response pattern plot to '{path}'")
        logger.info("\nResponse Length Summary:\n" + self.df[['high_len', 'low_len']].describe().to_string())
        logger.info("\nResponse Complexity Summary:\n" + self.df[['high_comp', 'low_comp']].describe().to_string())

    def _analyze_sentiment(self):
        logger.info("Analyzing linguistic markers (sentiment)...")
        sid = SentimentIntensityAnalyzer()
        self.df['high_sentiment'] = self.df['high_trait_response'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
        self.df['low_sentiment'] = self.df['low_trait_response'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

        df_melt = self.df.melt(id_vars=['trait_dimension'], value_vars=['high_sentiment', 'low_sentiment'], var_name='type', value_name='score')
        plt.figure(figsize=(15, 8))
        sns.violinplot(data=df_melt, x='trait_dimension', y='score', hue='type', split=True, palette='coolwarm')
        plt.title('Sentiment Score Distribution by Trait', fontsize=16); plt.axhline(0, c='k', ls='--')
        
        path = self.output_dir / "sentiment_distribution_plot.png"
        plt.savefig(path); plt.close()
        logger.info(f"  - Saved sentiment distribution plot to '{path}'")
        summary = self.df.groupby('trait_dimension')[['high_sentiment', 'low_sentiment']].agg(['mean', 'median'])
        logger.info("\nSentiment Score Summary:\n" + summary.to_string())

    def run_purity_and_contamination_analysis(self):
        """Runs vocabulary overlap and classifier-based purity assessment."""
        if self.df is None: self.load_and_prepare_data()
        if not self.trait_vocab: self._analyze_tfidf() # Dependency
        
        logger.info("\n[PHASE 3] Starting purity and contamination analysis...")
        self._analyze_vocabulary_overlap()
        self._run_purity_assessment()

    def _analyze_vocabulary_overlap(self):
        logger.info("Analyzing vocabulary overlap between all trait concepts...")
        labels = list(self.trait_vocab.keys())
        matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
        for i, l1 in enumerate(labels):
            for j, l2 in enumerate(labels):
                set1, set2 = self.trait_vocab[l1], self.trait_vocab[l2]
                union_len = len(set1.union(set2))
                matrix.iloc[i, j] = len(set1.intersection(set2)) / union_len if union_len > 0 else 0
        
        plt.figure(figsize=(12, 10)); sns.heatmap(matrix, annot=True, cmap='Reds', fmt=".2f")
        plt.title('Jaccard Similarity of Top Keywords', fontsize=16); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        path = self.output_dir / "vocabulary_overlap_heatmap.png"
        plt.savefig(path); plt.close()
        logger.info(f"  - Saved vocabulary overlap heatmap to '{path}'")
        logger.info("\nVocabulary Overlap Matrix (Jaccard Similarity):\n" + matrix.to_string())

    def _run_purity_assessment(self):
        logger.info(f"Running classifier-based purity assessment with threshold {self.args.purity_threshold:.2f}...")
        self._load_classifier()
        purity_dir = self.output_dir / "purified_subsets"
        purity_dir.mkdir(exist_ok=True)
        
        for trait in self.df['trait_dimension'].unique():
            logger.info(f"  - Purifying '{trait}' subset...")
            subset_df = self.df[self.df['trait_dimension'] == trait].copy()
            if subset_df.empty: continue
            
            predictions = self.classifier(subset_df['high_trait_response'].tolist(), batch_size=32, truncation=True)
            pure_indices = [
                subset_df.index[i] for i, p_list in enumerate(predictions)
                if {item['label']: item['score'] for item in p_list}.get(trait, 0) > self.args.purity_threshold
            ]
            purified_df = self.df.loc[pure_indices]
            
            retention = len(purified_df) / len(subset_df) if len(subset_df) > 0 else 0
            logger.info(f"    - Original size: {len(subset_df)}, Purified size: {len(purified_df)} ({retention:.2%})")
            
            output_path = purity_dir / f"purified_{trait}.csv"
            purified_df.to_csv(output_path, index=False)
            logger.info(f"    - Saved purified subset to '{output_path}'")

    def run_final_analysis(self):
        """Runs bias screening and generates heuristic-based candidate sets."""
        if self.df is None: self.load_and_prepare_data()
        
        logger.info("\n[PHASE 4] Starting bias screening and heuristic analysis...")
        sensitive_keywords = ['gender', 'race', 'religion', 'man', 'woman', 'male', 'female', 'he', 'she', 'stupid', 'idiot', 'hate']
        self.df['flag_for_review'] = self.df.apply(lambda row: any(k in str(row['high_trait_response']).lower().split() or k in str(row['low_trait_response']).lower().split() for k in sensitive_keywords), axis=1)
        logger.info(f"Found {self.df['flag_for_review'].sum()} rows flagged for manual safety/bias review.")

        logger.info("Identifying most contrastive pairs via heuristics...")
        if 'high_len' not in self.df.columns:
             self.df['high_len'] = self.df['high_trait_response'].str.split().str.len()
             self.df['low_len'] = self.df['low_trait_response'].str.split().str.len()
             sid = SentimentIntensityAnalyzer()
             self.df['high_sentiment'] = self.df['high_trait_response'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
             self.df['low_sentiment'] = self.df['low_trait_response'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
        
        self.df['contrast_score'] = (
            abs(self.df['low_len'] - self.df['high_len']) / self.df[['low_len', 'high_len']].max().max() +
            abs(self.df['low_sentiment'] - self.df['high_sentiment'])
        )
        top_contrastive = self.df.sort_values('contrast_score', ascending=False).head(500)
        path = self.output_dir / "heuristic_steering_vector_candidates.csv"
        top_contrastive.to_csv(path, index=False)
        logger.info(f"Exported top 500 heuristically contrastive pairs to '{path}'")

def main():
    parser = argparse.ArgumentParser(description="Run a comprehensive EDA on a contrastive personality dataset.")
    parser.add_argument("--input_files", nargs='+', required=True, help="Paths to one or more input CSV data files.")
    parser.add_argument("--output_dir", type=str, default="./eda_analysis_results", help="Directory to save all analysis outputs.")
    
    # Execution control
    parser.add_argument("--all", action="store_true", help="Run all analysis stages.")
    parser.add_argument("--linguistic", action="store_true", help="Run only the linguistic analysis (Phase 2).")
    parser.add_argument("--purity", action="store_true", help="Run only the purity and contamination analysis (Phase 3).")
    parser.add_argument("--final", action="store_true", help="Run only the final bias/heuristic analysis (Phase 4).")
    
    # Parameters
    parser.add_argument("--purity_threshold", type=float, default=0.50, help="Confidence threshold for classifier-based data purification.")
    
    args = parser.parse_args()
    
    # Default to running all if no specific stage is selected
    run_all = args.all or not any([args.linguistic, args.purity, args.final])
    
    analyzer = DatasetAnalyzer(args)
    start_time = time.time()
    
    if run_all or args.linguistic:
        analyzer.analyze_linguistic_features()
    
    if run_all or args.purity:
        analyzer.run_purity_and_contamination_analysis()

    if run_all or args.final:
        analyzer.run_final_analysis()

    total_time = time.time() - start_time
    logger.info(f"\n--- EDA COMPLETE ---")
    logger.info(f"Total runtime: {total_time:.2f} seconds.")
    logger.info(f"All outputs saved in: {analyzer.output_dir.resolve()}")

if __name__ == "__main__":
    # NLTK download check
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        print("Downloading NLTK vader_lexicon...")
        nltk.download('vader_lexicon')
    main()