# Personality Manipulation in LLMs against Benchmark Performance

This repository contains scripts for manipulating personality traits in large language models and evaluating their performance on various benchmarks.

## Overview

The project explores three methods for personality manipulation:
- **PEFT (Parameter-Efficient Fine-Tuning)**: Using LoRA adapters
- **Steering Vectors**: Direct activation steering
- **Prompting**: Context-based personality injection

Models are evaluated on three benchmarks:
- **MMLU**: Massive Multitask Language Understanding
- **GAIA**: General AI Assistant benchmark
- **BBQ**: Bias Benchmark for QA

## Scripts

### Core Evaluation Pipeline

#### `evaluate_generate.py`
Generates model responses for different personality traits across benchmarks.

**Usage:**
```bash
python evaluate_generate.py --model gemma2 --benchmark MMLU --method peft
python evaluate_generate.py --model llama3 --benchmark GAIA --method steering
python evaluate_generate.py --model gemma2 --benchmark BBQ --method prompting
```

**Parameters:**
- `--model`: Model to use (`gemma2` or `llama3`)
- `--benchmark`: Benchmark to run (`MMLU`, `GAIA`, `BBQ`)
- `--method`: Personality method (`peft`, `steering`, `prompting`)
- `--batch_size`: Override default batch size

#### `evaluate_score.py`
Scores and classifies generated responses using Azure OpenAI judge and personality classifier.

**Usage:**
```bash
python evaluate_score.py raw_outputs_gemma2_MMLU_peft.jsonl
```

### Data Analysis

#### `eda.py`
Comprehensive exploratory data analysis tool for personality datasets.

**Usage:**
```bash
python eda.py --all --input_files train.csv test.csv --output_dir ./analysis_results
python eda.py --linguistic --input_files data.csv --purity_threshold 0.6
```

**Features:**
- Word clouds and TF-IDF analysis
- Response pattern analysis (length, complexity)
- Sentiment analysis
- Vocabulary overlap detection
- Classifier-based data purification
- Bias screening and heuristic analysis

### Model Training

#### `generate_gemma_2_PEFT.py`
Trains LoRA adapters for Gemma-2 model on personality traits.

**Usage:**
```bash
python generate_gemma_2_PEFT.py --trait extraversion --epochs 3
python generate_gemma_2_PEFT.py --trait all --epochs 5
```

#### `generate_llama_3_PEFT.py`
Trains LoRA adapters for Llama-3 model on personality traits.

#### `generate_gemma_2_steering.py`
Extracts and calibrates steering vectors for Gemma-2 model.

**Usage:**
```bash
python generate_gemma_2_steering.py --all
python generate_gemma_2_steering.py --extract
python generate_gemma_2_steering.py --calibrate
```

### Data Generation

#### `personality_dataset_generation.py`
Generates personality contrastive datasets for training.

### Analysis Scripts

#### `calculate_stability_metrics.py`
Calculates normalized stability metrics for personality manipulation methods across benchmarks.

**Usage:**
```bash
python calculate_stability_metrics.py
python calculate_stability_metrics.py --debug
```

**Features:**
- Normalizes performance deltas across MMLU, GAIA, and BBQ benchmarks
- Calculates consistency (inverse of variance) and disruption metrics
- Provides composite stability scores for method comparison
- Aggregates results by method, personality, and combination levels
- Generates detailed analysis reports

**Output:**
- `normalized_stability_results.csv`: Aggregated stability rankings
- `detailed_normalized_stability.csv`: Detailed metrics for all combinations

#### `generate_alignment_numbers.ipynb`
Jupyter notebook for analyzing personality alignment metrics.

## Results

The `scripts/` folder contains CSV files with experimental results:
- `results_gemma2_*_peft.csv`: PEFT method results
- `results_gemma2_*_steering.csv`: Steering method results  
- `results_gemma2_*_prompting.csv`: Prompting method results
- `results_llama3_*_*.csv`: Llama-3 model results

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
HF_HOME=/path/to/huggingface/cache
TMPDIR=/path/to/temp/directory
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT=your_deployment
AZURE_OPENAI_SUBSCRIPTION_KEY=your_key
AZURE_OPENAI_API_VERSION=your_version
```

3. Run experiments:
```bash
# Generate responses
python scripts/evaluate_generate.py --model gemma2 --benchmark MMLU --method peft

# Score responses
python scripts/evaluate_score.py raw_outputs_gemma2_MMLU_peft.jsonl
```

## File Structure

```
scripts/
├── evaluate_generate.py      # Main generation pipeline
├── evaluate_score.py         # Scoring and classification
├── eda.py                    # Data analysis tool
├── calculate_stability_metrics.py # Stability analysis
├── generate_gemma_2_PEFT.py  # Gemma-2 PEFT training
├── generate_llama_3_PEFT.py  # Llama-3 PEFT training
├── generate_gemma_2_steering.py # Steering vector extraction
├── personality_dataset_generation.py # Dataset generation
├── generate_alignment_numbers.ipynb # Analysis notebook
└── results_*.csv            # Experimental results
```

## Dependencies

- PyTorch
- Transformers
- PEFT
- Datasets
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- WordCloud
- Azure OpenAI
- Dotenv
