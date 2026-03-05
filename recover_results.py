import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

def recover_results_from_incremental(output_dir: str):
    """Recover all results from incremental files"""
    output_path = Path(output_dir)
    
    # Find all incremental files
    incremental_files = list(output_path.glob("results_incremental_*.json"))
    
    if not incremental_files:
        print("No incremental files found!")
        return
    
    print(f"Found {len(incremental_files)} incremental files")
    
    all_results = []
    all_errors = []
    
    for file_path in sorted(incremental_files):
        print(f"Processing {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                # Don't skip corrupted files automatically, but provide
                # explicit information about which file failed and why so
                # it can be repaired or re-run using the checkpoint cleaner.
                print("\nERROR: Failed to parse incremental file as JSON:")
                print(f"  File : {file_path}")
                print(f"  Error: {e}")
                # Re-raise so the traceback is preserved after this context
                raise
            all_results.extend(data.get('results', []))
            all_errors.extend(data.get('errors', []))
    
    # Remove duplicates based on unique key
    seen_keys = set()
    unique_results = []
    
    for result in all_results:
        key = f"{result['model']}|{result['dataset']}|{result['prompt_type']}|{result['idx']}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique_results.append(result)
    
    print(f"Total unique results: {len(unique_results)}")
    print(f"Total errors: {len(all_errors)}")
    
    # Save recovered results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    recovered_file = output_path / f"results_recovered_{timestamp}.json"
    with open(recovered_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': unique_results,
            'errors': all_errors,
            'recovery_info': {
                'recovered_from': len(incremental_files),
                'total_results': len(unique_results),
                'total_errors': len(all_errors)
            }
        }, f, ensure_ascii=False, indent=2)
    
    # Save CSV
    if unique_results:
        df = pd.DataFrame(unique_results)
        csv_file = output_path / f"results_recovered_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Print summary by model
        print("\nResults by model:")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            correct = len(model_df[model_df['is_correct']])
            total = len(model_df)
            accuracy = correct / total if total > 0 else 0
            print(f"  {model}: {correct}/{total} ({accuracy:.2%})")

        # Print summary by dataset, preferring short dataset names from the data
        if 'source_dataset_short' in df.columns:
            dataset_col = 'source_dataset_short'
        elif 'source_dataset' in df.columns:
            dataset_col = 'source_dataset'
        else:
            dataset_col = 'dataset'

        print("\nResults by dataset (using column:", dataset_col, "):")
        for ds in df[dataset_col].dropna().unique():
            ds_df = df[df[dataset_col] == ds]
            correct = len(ds_df[ds_df['is_correct']])
            total = len(ds_df)
            accuracy = correct / total if total > 0 else 0
            print(f"  {ds}: {correct}/{total} ({accuracy:.2%})")
    
    print(f"\nRecovered results saved to:")
    print(f"  JSON: {recovered_file}")
    print(f"  CSV: {csv_file}")

if __name__ == "__main__":
    # Update this path to your actual output directory
    output_dir = "results/v5"  # Changed from "output" to "results"
    recover_results_from_incremental(output_dir)
