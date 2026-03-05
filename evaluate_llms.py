import re
import time
import requests
import numpy as np
import os
import pandas as pd
import ollama
from ollama import Client
import litellm
from litellm import completion
import json
import traceback
import yaml
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Union
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

### Model Client Managers ###

# Create a client manager to handle connection recycling for Ollama
class OllamaClientManager:
    def __init__(self, host='http://localhost:11434', timeout=30.0, max_requests=50):
        self.host = host
        self.timeout = timeout
        self.max_timeout = 180 # Maximum timeout to try
        self.max_requests = max_requests
        self.request_count = 0
        self._create_client()
        
    def _create_client(self):
        """Create a new client instance"""
        self.client = Client(host=self.host, timeout=self.timeout)
        self.request_count = 0
        logger.info(f"Created new Ollama client connection with {self.timeout}s timeout")
        
    def chat(self, **kwargs):
        """Chat with automatic connection recycling and adaptive timeout"""
        self.request_count += 1
        
        # Recycle connection periodically
        if self.request_count >= self.max_requests:
            logger.info(f"Recycling connection after {self.request_count} requests")
            time.sleep(1)
            self._create_client()
            
        try:
            return self.client.chat(**kwargs)
        except Exception as e:
            if "timed out" in str(e):
                # Increase timeout for next attempt
                if self.timeout < self.max_timeout:
                    self.timeout = min(self.timeout * 1.5, self.max_timeout)
                    logger.warning(f"Timeout detected, increasing to {self.timeout}s")
                    self._create_client()
                raise
            elif "Connection reset" in str(e) or "Connection refused" in str(e):
                logger.warning("Connection error detected, recreating client...")
                time.sleep(2)
                self._create_client()
                # Retry once with new connection
                return self.client.chat(**kwargs)
            raise

# LiteLLM client manager for API models
class LiteLLMClientManager:
    def __init__(self):
        self.total_cost = 0.0
        self.request_count = 0
        
    def chat(self, model: str, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 512) -> Dict:
        """Chat using LiteLLM with cost tracking"""
        self.request_count += 1
        
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract cost if available
            cost = 0.0
            try:
                if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                    cost = response._hidden_params.get('response_cost', 0.0)
                    if cost is None:
                        cost = 0.0
            except Exception as cost_error:
                logger.debug(f"Cost calculation error: {cost_error}")
                cost = 0.0
            
            if isinstance(cost, (int, float)):
                self.total_cost += cost
            
            # Format response to match Ollama structure
            return {
                'message': {
                    'content': response.choices[0].message.content
                },
                'cost': cost
            }
            
        except Exception as e:
            logger.error(f"LiteLLM error: {type(e).__name__}: {str(e)}")
            raise

# Initialize client managers
ollama_manager = OllamaClientManager(timeout=60.0, max_requests=40)
litellm_manager = LiteLLMClientManager()

### Configuration ###

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

### Simple Response Parser ###

def parse_single_choice_response(response_text: str) -> Optional[str]:
    """
    Parse single letter response (a, b, c, d, or e).
    Returns the letter if found, None otherwise.
    """
    # Clean the response
    response_text = response_text.strip().lower()
    
    # Look for single letter at the start or end
    # First check if the entire response is just a single letter
    if response_text in ['a', 'b', 'c', 'd', 'e']:
        return response_text
    
    # Look for patterns like "a.", "a)", "(a)", "a:", etc.
    patterns = [
        r'^([a-e])[\.:\)\s]',  # Start with letter followed by punctuation
        r'\(([a-e])\)',         # Letter in parentheses
        r'([a-e])\s*$',         # Letter at the end
        r'^svar:\s*([a-e])',    # "svar: a" format
        r'bogstav[et]*\s+([a-e])', # "bogstav a" or "bogstavet a"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # If no pattern matched, look for any occurrence of a-e
    # but only if there's exactly one such letter in the response
    letters_found = re.findall(r'[a-e]', response_text)
    if len(letters_found) == 1:
        return letters_found[0]
    
    return None

# Add a letter-to-exp mapping function
def letter_to_exp(letter: str) -> str:
    """Convert letter response to exp format"""
    mapping = {
        'a': 'exp1', 'A': 'exp1',
        'b': 'exp2', 'B': 'exp2',
        'c': 'exp3', 'C': 'exp3',
        'd': 'exp4', 'D': 'exp4',
        'e': 'dont_know', 'E': 'dont_know'
    }
    return mapping.get(letter, letter)

@dataclass
class ExperimentMetadata:
    """Metadata for the experiment run"""
    experiment_name: str
    timestamp: str
    models: List[str]
    datasets: List[Dict]
    prompt_types: List[str]
    config_hash: str
    total_api_cost: float = 0.0
    
### Experiment Tracking ###

class ExperimentTracker:
    """Tracks experiment progress and allows resumption"""
    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.checkpoint_file = self.output_dir / f"{experiment_name}_checkpoint.json"
        self.processed_items = set()
        self._load_processed_items()
        
    def _load_processed_items(self):
        """Load previously processed items"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.processed_items = set(data.get('processed_items', []))
    
    def get_item_key(self, model: str, dataset: str, prompt_type: str, idx: int) -> str:
        """Generate unique key for an item"""
        return f"{model}|{dataset}|{prompt_type}|{idx}"
    
    def is_processed(self, model: str, dataset: str, prompt_type: str, idx: int) -> bool:
        """Check if item has been processed"""
        key = self.get_item_key(model, dataset, prompt_type, idx)
        return key in self.processed_items
    
    def mark_processed(self, model: str, dataset: str, prompt_type: str, idx: int):
        """Mark item as processed"""
        key = self.get_item_key(model, dataset, prompt_type, idx)
        self.processed_items.add(key)
        self._save_checkpoint()
    
    def _save_checkpoint(self):
        """Save checkpoint to disk"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'processed_items': list(self.processed_items),
                'last_updated': datetime.now().isoformat()
            }, f)

### Results Management ###

class ResultsManager:
    """Manages experiment results with structured output"""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_buffer = []
        self.errors_buffer = []
        self.all_results = []  # Keep all results across models
        self.all_errors = []   # Keep all errors across models

    def _sanitize_record(self, record: Dict) -> Dict:
        """Ensure all values in a record are JSON-serializable.

        This is defensive against unexpected objects (e.g. exceptions) ending up
        in fields like 'error'. Non-primitive types are converted to strings,
        while dicts/lists are sanitized recursively.
        """

        def _convert(key, value):
            # Log detailed info for unexpected error payloads
            if key == 'error' and value is not None and not isinstance(value, (str, int, float, bool)):
                logger.debug(
                    "Sanitizing non-primitive error value of type %s: %s",
                    type(value), repr(value)
                )

            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, dict):
                return {k: _convert(k, v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(key, v) for v in value]
            # Fallback for anything else (e.g. Exception, custom objects)
            return str(value)

        return {k: _convert(k, v) for k, v in record.items()}

    def add_result(self, result: Dict):
        """Add a result to the buffer"""
        sanitized = self._sanitize_record(result)
        self.results_buffer.append(sanitized)
        self.all_results.append(sanitized)  # Also add to permanent storage
        
        # Save incrementally every 100 results
        if len(self.results_buffer) % 100 == 0:
            self._save_incremental()
    
    def add_error(self, error: Dict):
        """Add an error to the buffer"""
        sanitized = self._sanitize_record(error)
        self.errors_buffer.append(sanitized)
        self.all_errors.append(sanitized)  # Also add to permanent storage
    
    def _save_incremental(self):
        """Save results incrementally"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        incremental_file = self.output_dir / f"results_incremental_{timestamp}.json"
        
        with open(incremental_file, 'w', encoding='utf-8') as f:
            try:
                json.dump({
                    'results': self.results_buffer,
                    'errors': self.errors_buffer
                }, f, ensure_ascii=False, indent=2)
            except TypeError as e:
                # If incremental save fails due to an unexpected non-serializable value,
                # log the issue and continue the experiment. Final saving will still run
                # using the accumulated all_results/all_errors.
                logger.error(f"Failed incremental save to {incremental_file}: {e}. Skipping this incremental write.")
    
    def save_final_results(self, metadata: ExperimentMetadata) -> Tuple[Path, Path]:
        """Save final results with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results - use all_results instead of results_buffer
        results_file = self.output_dir / f"results_final_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': metadata.__dict__,
                'results': self.all_results,  # Changed from results_buffer
                'errors': self.all_errors,    # Changed from errors_buffer
                'summary': self._calculate_summary()
            }, f, ensure_ascii=False, indent=2)
        
        # Save CSV for easy analysis
        if self.all_results:  # Changed from results_buffer
            df = pd.DataFrame(self.all_results)
            csv_file = self.output_dir / f"results_final_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
        else:
            csv_file = None
            
        logger.info(f"Saved results to {results_file}")
        if csv_file:
            logger.info(f"Saved CSV to {csv_file}")
            
        return results_file, csv_file
    
    def _calculate_summary(self) -> Dict:
        """Calculate summary statistics"""
        if not self.all_results:  # Changed from results_buffer
            return {}
            
        df = pd.DataFrame(self.all_results)  # Changed from results_buffer
        summary = {
            'total_evaluations': len(self.all_results),  # Changed from results_buffer
            'total_errors': len(self.all_errors),        # Changed from errors_buffer
            'by_model': {},
            'by_dataset': {},   # Uses source_dataset_short/source_dataset where available
            'by_prompt_type': {}
        }
        
        # Accuracy by model
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            summary['by_model'][model] = {
                'total': len(model_df),
                'correct': len(model_df[model_df['is_correct']]),
                'accuracy': len(model_df[model_df['is_correct']]) / len(model_df) if len(model_df) > 0 else 0
            }

        # Accuracy by underlying dataset (v5 uses a single column: source_dataset)
        if 'source_dataset' in df.columns:
            dataset_col = 'source_dataset'
        else:
            dataset_col = 'dataset'

        for dataset in df[dataset_col].unique():
            dataset_df = df[df[dataset_col] == dataset]
            summary['by_dataset'][dataset] = {
                'total': len(dataset_df),
                'correct': len(dataset_df[dataset_df['is_correct']]),
                'accuracy': len(dataset_df[dataset_df['is_correct']]) / len(dataset_df) if len(dataset_df) > 0 else 0
            }
            
        return summary

### Prompt Formatting ###

class PromptFormatter:
    """Handles prompt formatting based on config"""
    def __init__(self, config: Dict):
        self.config = config
        self.prompts = config['prompts']
    
    def format_prompt(self, prompt_type: str, data: Dict) -> str:
        """Format prompt with data"""
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        prompt_config = self.prompts[prompt_type]
        
        # Get prefix if exists
        prefix = prompt_config.get('prefix', '')
        
        # Format template
        template = prompt_config['template']
        formatted = template.format(**data)
        
        return prefix + formatted

### LLM Evaluation ###

class LLMEvaluator:
    """Handles LLM evaluation with proper error handling for both Ollama and LiteLLM"""
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature
    
    def _is_litellm_model(self, model: str) -> bool:
        """Check if model should use LiteLLM"""
        return model.startswith(('openrouter/', 'anthropic/', 'openai/', 'gpt-', 'claude-'))
    
    def evaluate(self, model: str, prompt: str, max_retries: int = 3) -> Tuple[Optional[str], Optional[str], float]:
        """Evaluate prompt with LLM and return parsed response letter and cost"""
        cost = 0.0
        
        for attempt in range(max_retries):
            try:
                # Choose appropriate client
                if self._is_litellm_model(model):
                    # Use LiteLLM
                    response = litellm_manager.chat(
                        model=model,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=self.temperature,
                        max_tokens=512
                    )
                    cost = response.get('cost', 0.0)
                else:
                    # Use Ollama
                    response = ollama_manager.chat(
                        model=model,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={'temperature': self.temperature, 'num_predict': 512}
                    )
                
                response_text = response['message']['content']
                
                # Parse the single letter response
                parsed_letter = parse_single_choice_response(response_text)
                
                if parsed_letter:
                    return parsed_letter, None, cost
                else:
                    error = f"Could not parse response: {response_text[:100]}..."
                    logger.warning(f"Attempt {attempt + 1} failed: {error}")
                    
            except Exception as e:
                error = f"Unexpected error: {str(e)}"
                if "Connection reset by peer" in str(e) or "Connection refused" in str(e):
                    logger.warning("Retrying due to connection issue")
                    time.sleep(5) # Wait for 5 seconds and retry
                
                logger.error(f"Attempt {attempt + 1} failed: {error}")
                if attempt < max_retries - 1:
                    continue
        
        return None, error, cost

### Main Experiment Runner ###

def shuffle_explanations(row: pd.Series, seed: int = None) -> Tuple[Dict, Dict]:
    """Shuffle explanations and return mapping"""
    if seed is not None:
        np.random.seed(seed)
    
    # Original explanations with their labels
    original_explanations = {
        'exp1': row['exp1'],
        'exp2': row['exp2'], 
        'exp3': row['exp3'],
        'exp4': row['exp4']
    }
    
    # Create list of (original_key, explanation) pairs
    exp_pairs = list(original_explanations.items())
    
    # Shuffle the pairs
    np.random.shuffle(exp_pairs)
    
    # Create shuffled dict and mapping
    shuffled = {}
    reverse_mapping = {}  # Maps from shuffled position to original position
    
    for i, (original_key, explanation) in enumerate(exp_pairs):
        shuffled_key = f'exp{i+1}'
        shuffled[shuffled_key] = explanation
        reverse_mapping[shuffled_key] = original_key
    
    return shuffled, reverse_mapping

def run_experiment(config_path: str = "config.yaml"):
    """Run the full experiment"""
    # Check for API keys if using LiteLLM models
    config = load_config(config_path)
    litellm_models = [m for m in config['models'] if m.startswith(('openrouter/', 'anthropic/', 'openai/', 'gpt-', 'claude-'))]
    
    if litellm_models and "OPENROUTER_API_KEY" not in os.environ:
        logger.error("OPENROUTER_API_KEY environment variable not set!")
        logger.info("Set it with: export OPENROUTER_API_KEY='sk-or-v1-...'")
        return None, None
    
    # Calculate config hash for tracking
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Initialize components
    output_dir = config['experiment']['output_dir']
    experiment_name = config['experiment']['name']
    
    tracker = ExperimentTracker(output_dir, experiment_name)
    results_manager = ResultsManager(output_dir)
    formatter = PromptFormatter(config)
    evaluator = LLMEvaluator(temperature=config.get('temperature', 0.1))
    
    # Create metadata
    metadata = ExperimentMetadata(
        experiment_name=experiment_name,
        timestamp=datetime.now().isoformat(),
        models=config['models'],
        datasets=config['datasets'],
        prompt_types=list(config['prompts'].keys()),
        config_hash=config_hash,
        total_api_cost=0.0
    )
    
    # Main experiment loop
    total_items = len(config['models']) * len(config['datasets']) * len(config['prompts'])
    
    with tqdm(total=total_items, desc="Running experiments") as pbar:
        for model in config['models']:
            logger.info(f"\nProcessing model: {model}")
            
            for dataset_config in config['datasets']:
                dataset_name = dataset_config['name']
                
                # Load dataset
                df = pd.read_csv(dataset_config['file_path'], sep='\t')

                # Log underlying sources present in this v5 dataset (based on 'source' column)
                if 'source' in df.columns:
                    underlying = sorted(df['source'].dropna().unique().tolist())
                    logger.info(f"Processing dataset group: {dataset_name} (sources: {underlying})")
                else:
                    logger.info(f"Processing dataset: {dataset_name}")
                
                # Drop rows with missing explanations
                df.dropna(subset=['exp1', 'exp2', 'exp3', 'exp4'], inplace=True)
                
                for prompt_type in config['prompts']:
                    for idx, row in df.iterrows():
                        # Check if already processed
                        if tracker.is_processed(model, dataset_name, prompt_type, idx):
                            pbar.update(1)
                            continue
                        
                        # Prepare data
                        if dataset_config.get('shuffle_explanations', True):
                            seed = config['experiment'].get('random_seed', 42) + idx
                            shuffled_exps, mapping = shuffle_explanations(row, seed)
                        else:
                            shuffled_exps = {
                                'exp1': row['exp1'],
                                'exp2': row['exp2'],
                                'exp3': row['exp3'],
                                'exp4': row['exp4']
                            }
                            mapping = {f'exp{i}': f'exp{i}' for i in range(1, 5)}
                        
                        # Format prompt (v5: 'word' column replaces 'lemma')
                        prompt_data = {
                            'lemma': row['word'],
                            'sentence': row['sentence'],
                            **shuffled_exps
                        }
                        
                        prompt = formatter.format_prompt(prompt_type, prompt_data)
                        
                        # Evaluate
                        start_time = datetime.now()
                        response_letter, error, cost = evaluator.evaluate(model, prompt)
                        end_time = datetime.now()
                        
                        # Update total cost
                        metadata.total_api_cost += cost
                        
                        # Add a small delay to prevent overwhelming the server
                        time.sleep(0.5)
                        
                        # Record result (v5 schema: 'word' and single underlying dataset column 'source_dataset')
                        result = {
                            'model': model,
                            # High-level dataset label from config
                            'dataset': dataset_name,
                            # Source dataset name from v5 TSV
                            'source_dataset': row.get('source'),
                            'prompt_type': prompt_type,
                            'idx': idx,
                            'lemma': row['word'],
                            'sentence': row['sentence'],
                            'correct_answer': row['exp1'], # Always use exp1 for correct answer
                            'timestamp': start_time.isoformat(),
                            'response_time': (end_time - start_time).total_seconds(),
                            'api_cost': cost
                        }
                        
                        if response_letter:
                            # Convert letter to exp format
                            predicted_position = letter_to_exp(response_letter)
                            
                            # Handle "don't know" response
                            if predicted_position == 'dont_know':
                                is_correct = False
                                predicted_original = 'dont_know'
                            else:
                                # Map back to original using reverse mapping
                                predicted_original = mapping.get(predicted_position, predicted_position)
                                # Check if correct (exp1 is always the correct answer in original data)
                                is_correct = predicted_original == 'exp1'
                            
                            result.update({
                                'predicted_letter': response_letter,
                                'predicted_position': predicted_position,  # The exp format (shuffled position)
                                'predicted_original': predicted_original, # Original position
                                'is_correct': is_correct,
                                'chose_dont_know': response_letter == 'e',
                                'error': None,
                                'shuffle_mapping': mapping  # Optional: store mapping for debugging
                            })
                            
                            results_manager.add_result(result)
                        else:
                            result.update({
                                'predicted_letter': None,
                                'predicted_position': None,
                                'predicted_original': None,
                                'is_correct': False,
                                'chose_dont_know': False,
                                'error': error
                            })
                            
                            results_manager.add_error(result)
                        
                        # Mark as processed
                        tracker.mark_processed(model, dataset_name, prompt_type, idx)
                        pbar.update(1)
    
    # Save final results
    results_file, csv_file = results_manager.save_final_results(metadata)
    
    logger.info("\nExperiment completed!")
    logger.info(f"Total API cost: ${metadata.total_api_cost:.4f}")
    
    return results_file, csv_file

if __name__ == "__main__":
    run_experiment()
