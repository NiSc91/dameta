import json
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml

matplotlib.use('Agg')

class ErrorAnalyzer:
    """Analyze errors from LLM evaluation results"""
    
    def __init__(self, results_file: str, config_file: str = None):
        """Initialize with results file and optional config file"""
        self.results_file = Path(results_file)
        self.config_file = Path(config_file) if config_file else None
        self.data = self._load_results()
        self.df = pd.DataFrame(self.data['results'])
        # For v4, we derive metadata directly from the results dataframe, so
        # we always attempt to load it regardless of whether a config file is
        # provided. The config file is only needed for optional extras.
        self.dataset_metadata = self._load_dataset_metadata()
    
    def _load_results(self) -> Dict:
        """Load results from JSON file"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_dataset_metadata(self) -> Dict:
        """Load metadata for samples"""
        TOPIC_TRANSLATIONS = {
            'OB farve': 'OB color', 'OB følelse': 'OB emotion',
            'OB rejse': 'OB travel', 'OB smag': 'OB taste',
            'anatomi': 'anatomy', 'arkitektur': 'architecture',
            'byggeri': 'construction', 'erhverv': 'business',
            'familie': 'family', 'film': 'film',
            'genprox bog': 'genprox book', 'håndarbejde': 'handicraft',
            'håndværk': 'craftsmanship', 'kommunikation': 'communication',
            'litteratur': 'literature', 'mad/gastronomi': 'food/gastronomy',
            'medicin/sundhed': 'medicine/health', 'meteorologi': 'meteorology',
            'militær': 'military', 'musik': 'music',
            'psykologi': 'psychology', 'trafik': 'traffic',
            'skib/søfart': 'shipping/nautical'
        }

        metadata: Dict[str, Dict] = {}

        if 'source_dataset_short' not in self.df.columns:
            return metadata

        # Locate the combined v4 dataset TSV: prefer config.yaml if given,
        # otherwise fall back to the default path used in this project.
        combined_path: Path = None
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                for ds_cfg in config.get('datasets', []):
                    if ds_cfg.get('name') == 'combined_v4' and 'file_path' in ds_cfg:
                        candidate = Path(ds_cfg['file_path'])
                        if candidate.exists():
                            combined_path = candidate
                            break
            except Exception as e:
                print(f"Warning: Could not read config file for metadata: {e}")

        if combined_path is None:
            default_path = Path(__file__).parent / 'data' / 'v4' / 'combined_v4.tsv'
            if default_path.exists():
                combined_path = default_path

        if combined_path is None or not combined_path.exists():
            return metadata

        try:
            combined_df = pd.read_csv(combined_path, sep='\t')
        except Exception as e:
            print(f"Warning: Could not load combined v4 dataset for metadata: {e}")
            return metadata

        if 'dataset_short' not in combined_df.columns:
            return metadata

        df_results = self.df

        # Collect evaluated indices per original dataset (short code)
        evaluated_indices: Dict[str, set] = {}
        for ds in df_results['source_dataset_short'].dropna().unique():
            ds_df = df_results[df_results['source_dataset_short'] == ds]
            if len(ds_df) > 0:
                evaluated_indices[ds] = set(ds_df['idx'].unique())

        # 1) NS_DaFig: real type column exists in the original data
        if 'NS_DaFig' in evaluated_indices and 'type' in combined_df.columns:
            type_mapping: Dict[int, str] = {}
            for idx in evaluated_indices['NS_DaFig']:
                if 0 <= idx < len(combined_df):
                    row = combined_df.iloc[idx]
                    if row.get('dataset_short') == 'NS_DaFig' and pd.notna(row['type']):
                        raw_val = row['type']
                        # Treat negative values as invalid and skip them entirely
                        try:
                            if float(raw_val) < 0:
                                continue
                        except Exception:
                            # If it cannot be parsed as float, fall back to string check
                            if str(raw_val).strip().startswith('-'):
                                continue
                        type_value = str(int(float(raw_val))) if isinstance(raw_val, (int, float, np.number)) else str(raw_val)
                        type_mapping[idx] = type_value
            metadata['NS_DaFig'] = {'type_mapping': type_mapping}

        # 2) SN_DDO (non ad-hoc): always Type 1, with topics from `emne` if present
        if 'SN_DDO' in evaluated_indices:
            idxs = evaluated_indices['SN_DDO']
            type_mapping = {idx: '1' for idx in idxs}

            topic_mapping_english: Dict[int, str] = {}
            if 'emne' in combined_df.columns:
                for idx in idxs:
                    if 0 <= idx < len(combined_df):
                        row = combined_df.iloc[idx]
                        if row.get('dataset_short') == 'SN_DDO' and pd.notna(row['emne']):
                            danish_topic = row['emne']
                            english_topic = TOPIC_TRANSLATIONS.get(danish_topic, danish_topic)
                            topic_mapping_english[idx] = english_topic

            entry: Dict[str, Dict] = {'type_mapping': type_mapping}
            if topic_mapping_english:
                entry['topic_mapping'] = topic_mapping_english
            metadata['SN_DDO'] = entry

        # 3) Ad-hoc datasets: BSP_pol_ad_hoc and SN_DDO_ad_hoc are always Type 3
        for ds_name in ['BSP_pol_ad_hoc', 'SN_DDO_ad_hoc']:
            if ds_name in evaluated_indices:
                idxs = evaluated_indices[ds_name]
                type_mapping = {idx: '3' for idx in idxs}
                metadata[ds_name] = {'type_mapping': type_mapping}

        return metadata

    def create_all_figures(self, output_dir: str = "plots"):
        """Create all 5 required figures"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Figure 1: Overall comparison with human performance
        self._create_figure1_overall_comparison(output_path)
        
        # Figure 2 & 3: Distractor analysis per model for each prompt version
        #self._create_figure2_3_distractor_per_model(output_path)
        
        # Table
        self._create_table2_model_performance(output_path)
        
        # Figure 4: Error distribution by type (averaged over prompts)
        self._create_figure2_type_analysis(output_path)
        
        # Figure 3: Error distribution by topic (averaged over prompts)
        self._create_figure3_topic_analysis(output_path)
        
        # Save machine-readable summary of all key analyses
        self._save_analysis_summary(output_path)

        print(f"All figures saved to {output_path}")

    def _create_figure1_overall_comparison(self, output_path):
        """Figure 1: Overall LLM vs Human performance comparison with proprietary/local separation"""

        fig, ax = plt.subplots()

        # Define model categories
        proprietary_models = ['openrouter/openai/gpt-4o-mini', 'openrouter/anthropic/claude-3.5-sonnet']
        local_models = ['llama3.1', 'gemma2', 'mistral', 'qwen2.5', 'phi4']

        # Calculate overall statistics
        prompt_v1 = self.df[self.df['prompt_type'] == 'met_v1']
        prompt_v2 = self.df[self.df['prompt_type'] == 'met_v2']

        # Calculate accuracy for proprietary vs local models
        v1_proprietary = prompt_v1[prompt_v1['model'].isin(proprietary_models)]
        v1_local = prompt_v1[prompt_v1['model'].isin(local_models)]
        v2_proprietary = prompt_v2[prompt_v2['model'].isin(proprietary_models)]
        v2_local = prompt_v2[prompt_v2['model'].isin(local_models)]

        # Accuracy data
        human_acc = 89.58
        v1_prop_acc = v1_proprietary['is_correct'].mean() * 100 if len(v1_proprietary) > 0 else 0
        v1_local_acc = v1_local['is_correct'].mean() * 100 if len(v1_local) > 0 else 0
        v2_prop_acc = v2_proprietary['is_correct'].mean() * 100 if len(v2_proprietary) > 0 else 0
        v2_local_acc = v2_local['is_correct'].mean() * 100 if len(v2_local) > 0 else 0

        # Don't know rates
        v1_prop_dk = v1_proprietary[
                         'chose_dont_know'].mean() * 100 if 'chose_dont_know' in v1_proprietary.columns and len(
            v1_proprietary) > 0 else 0
        v1_local_dk = v1_local['chose_dont_know'].mean() * 100 if 'chose_dont_know' in v1_local.columns and len(
            v1_local) > 0 else 0
        v2_prop_dk = v2_proprietary[
                         'chose_dont_know'].mean() * 100 if 'chose_dont_know' in v2_proprietary.columns and len(
            v2_proprietary) > 0 else 0
        v2_local_dk = v2_local['chose_dont_know'].mean() * 100 if 'chose_dont_know' in v2_local.columns and len(
            v2_local) > 0 else 0

        # Left plot: Accuracy comparison
        categories = ['Proprietary LLMs', 'Open LLMs']
        v1_accuracies = [v1_prop_acc, v1_local_acc]
        v2_accuracies = [v2_prop_acc, v2_local_acc]

        x = np.arange(len(categories))
        width = 0.35

        colours_V1 = ['#0B5355', '#CB582B']
        colours_V2 = ['#0EA2AA', '#F79C61']
        human_colour = ['#EBBC21', '#BB9311']

        ax.axhline(y=human_acc, color=human_colour[0], linestyle='--', alpha=1,
                   linewidth=2.5)

        # Add label to the special bar
        ax.text(1.7, human_acc - 5,
                f'Human Accuracy: {human_acc:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=human_colour[1])

        # Create bars for both prompt versions
        bars1 = ax.bar(x + 1 - width / 2, v1_accuracies, width, label='Prompt V1',
                       color=colours_V1, alpha=0.8, )  # edgecolor='grey', linewidth=1)
        bars2 = ax.bar(x + 1 + width / 2, v2_accuracies, width, label='Prompt V2',
                       color=colours_V2, alpha=0.8, )  # edgecolor='grey', linewidth=1)

        # Add value labels on bars
        for bars, accs in [(bars1, v1_accuracies), (bars2, v2_accuracies)]:
            for bar, acc in zip(bars, accs):
                if acc > 0:  # Only show if there's data
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                            f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        # plt.title('Overall Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_xticks([1, 2], labels=categories)

        colours = [colours_V1[0], colours_V2[0], colours_V1[1], colours_V2[1]]
        labels = [
            'Proprietary (prompt 1)', 'Proprietary (prompt 2)', 'Open (prompt 1)',
            'Open (prompt 2)']

        # Create legend handles
        legend_elements = [Patch(facecolor=col, edgecolor='grey', label=lab)
                           for col, lab in zip(colours, labels)]

        # Add legend to plot
        ax.legend(handles=legend_elements, loc='lower center', ncol=2,
                  bbox_to_anchor=(0.5, 1.05)
                  )

        # Add extra space above the plot to avoid overlap
        fig.subplots_adjust(top=0.5)  # Adjust as needed

        ax.grid(axis='y', alpha=0.3, )

        fig.tight_layout()
        plt.savefig(output_path / 'figure1_overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 6))

        # Right plot: Distractor distribution
        display_labels = ['Literal distract.', 'Figurative distract.', 'Contradictory distract.', "Don't know"]

        # Human distribution
        human_dist = [3.70, 5.32, 1.39, 0]

        # Calculate distributions for proprietary and local models
        def calculate_distribution(df_subset):
            dist = []
            if len(df_subset) == 0:
                return [0, 0, 0]
            for exp in ['exp2', 'exp3', 'exp4', 'dont_know']:
                pred_count = (df_subset['predicted_original'] == exp).sum()
                dist.append(pred_count / len(df_subset) * 100)
            return dist

        v1_prop_dist = calculate_distribution(v1_proprietary)
        v1_local_dist = calculate_distribution(v1_local)
        v2_prop_dist = calculate_distribution(v2_proprietary)
        v2_local_dist = calculate_distribution(v2_local)

        x = np.arange(len(display_labels))
        width = 0.15

        # Create grouped bars
        bars1 = plt.bar(x - 1.5 * width, human_dist, width, label='Human', color=human_colour[0], alpha=0.8)
        bars2 = plt.bar(x - 0.5 * width, v1_prop_dist, width, label='Proprietary (prompt 1)', color='#0B5355',
                        alpha=0.8)
        bars3 = plt.bar(x + 0.5 * width, v2_prop_dist, width, label='Proprietary (prompt 2)', color='#0EA2AA',
                        alpha=0.8)
        bars4 = plt.bar(x + 1.5 * width, v1_local_dist, width, label='Open (prompt 1)', color='#CB582B', alpha=0.8)
        bars5 = plt.bar(x + 2.5 * width, v2_local_dist, width, label='Open (prompt 2)', color='#F79C61', alpha=0.8)

        plt.xlabel('Response Type', fontsize=12, fontweight='bold')
        plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        # plt.title('Response Distribution Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, labels=display_labels)
        plt.legend(loc='upper right', fontsize=12)

        plt.yticks(range(0, 13, 1))
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'figure2_distractor_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_table2_model_performance(self, output_path):
        """Table 2: Model performance comparison showing accuracy and don't-know rates"""
        
        # Prepare data for each model and prompt version
        models = sorted(self.df['model'].unique())
        
        # Define model categories for better organization
        proprietary_models = ['openrouter/openai/gpt-4o-mini', 'openrouter/anthropic/claude-3.5-sonnet']
        local_models = ['llama3.1', 'gemma2', 'mistral', 'qwen2.5', 'phi4']
        
        # Create table data
        table_data = []
        
        for model in models:
            # Get data for both prompt versions
            v1_df = self.df[(self.df['model'] == model) & (self.df['prompt_type'] == 'met_v1')]
            v2_df = self.df[(self.df['model'] == model) & (self.df['prompt_type'] == 'met_v2')]
            
            # Calculate metrics
            v1_accuracy = v1_df['is_correct'].mean() * 100 if len(v1_df) > 0 else 0
            v2_accuracy = v2_df['is_correct'].mean() * 100 if len(v2_df) > 0 else 0
            
            # Calculate don't-know rates if the column exists
            v1_dk = v1_df['chose_dont_know'].mean() * 100 if 'chose_dont_know' in v1_df.columns and len(v1_df) > 0 else 0
            v2_dk = v2_df['chose_dont_know'].mean() * 100 if 'chose_dont_know' in v2_df.columns and len(v2_df) > 0 else 0
            
            # Determine model type
            model_type = 'Proprietary' if model in proprietary_models else 'Local'
            
            table_data.append({
                'Model': model,
                'Type': model_type,
                'V1 Accuracy': v1_accuracy,
                'V1 Don\'t Know': v1_dk,
                'V2 Accuracy': v2_accuracy,
                'V2 Don\'t Know': v2_dk
            })
        
        # Sort by type (Proprietary first) then by V2 accuracy
        table_data.sort(key=lambda x: (x['Type'] != 'Proprietary', -x['V2 Accuracy']))
        
        # Create LaTeX table
        latex_table = []
        latex_table.append("\\begin{table}[h]")
        latex_table.append("\\centering")
        latex_table.append("\\caption{Model Performance Comparison: Accuracy and Don't-Know Rates}")
        latex_table.append("\\label{tab:model_performance}")
        latex_table.append("\\begin{tabular}{llrrrr}")
        latex_table.append("\\toprule")
        latex_table.append("\\multirow{2}{*}{Model} & \\multirow{2}{*}{Type} & \\multicolumn{2}{c}{Prompt V1} & \\multicolumn{2}{c}{Prompt V2} \\\\")
        latex_table.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}")
        latex_table.append(" & & Accuracy (\\%) & Don't Know (\\%) & Accuracy (\\%) & Don't Know (\\%) \\\\")
        latex_table.append("\\midrule")
        
        # Add human baseline
        latex_table.append(f"Human & - & 89.58 & - & 89.58 & - \\\\")
        latex_table.append("\\midrule")
        
        # Add model data
        current_type = None
        for row in table_data:
            # Add separator between proprietary and local models
            if current_type and current_type != row['Type']:
                latex_table.append("\\midrule")
            current_type = row['Type']
            
            # Format model name for LaTeX (escape underscores)
            model_name = row['Model'].replace('_', '\\_')
            
            # Format the row
            latex_table.append(f"""{model_name} & {row['Type']} & """
                            f"""{row['V1 Accuracy']:.1f} & {row["V1 Don't Know"]:.1f} & """
                            f"""{row['V2 Accuracy']:.1f} & {row["V2 Don't Know"]:.1f} \\\\""")

        latex_table.append("\\bottomrule")
        latex_table.append("\\end{tabular}")
        latex_table.append("\\end{table}")
        
        # Save LaTeX table to file
        latex_output = '\n'.join(latex_table)
        with open(output_path / 'table2_model_performance.tex', 'w') as f:
            f.write(latex_output)
        
        # Also create a CSV version for easy viewing
        import pandas as pd
        df_table = pd.DataFrame(table_data)
        df_table.to_csv(output_path / 'table2_model_performance.csv', index=False)
        
        # Print the table for console viewing
        print("\n" + "="*80)
        print("Table 2: Model Performance Comparison")
        print("="*80)
        print(f"{'Model':<40} {'Type':<12} {'V1 Acc%':>8} {'V1 DK%':>8} {'V2 Acc%':>8} {'V2 DK%':>8}")
        print("-"*80)
        print(f"{'Human':<40} {'-':<12} {89.58:>8.1f} {'-':>8} {89.58:>8.1f} {'-':>8}")
        print("-"*80)
        
        for row in table_data:
            print(f"""{row['Model']:<40} {row['Type']:<12} """
                f"""{row['V1 Accuracy']:>8.1f} {row["V1 Don't Know"]:>8.1f} """
                f"""{row['V2 Accuracy']:>8.1f} {row["V2 Don't Know"]:>8.1f}""")
        
        print("="*80)
        print(f"\nLaTeX table saved to: {output_path / 'table2_model_performance.tex'}")
        print(f"CSV table saved to: {output_path / 'table2_model_performance.csv'}")
    
    def _create_figure2_3_distractor_per_model(self, output_path):
        """Figures 2 & 3: Distractor analysis per model for each prompt version"""
        
        for prompt_idx, prompt_type in enumerate(['met_v1', 'met_v2'], 2):
            prompt_df = self.df[self.df['prompt_type'] == prompt_type]
            
            # Get unique models
            models = sorted(prompt_df['model'].unique())
            
            # Prepare data for each model
            model_data = {}
            for model in models:
                model_df = prompt_df[prompt_df['model'] == model]
                dist = []
                for exp in ['exp1', 'exp2', 'exp3', 'exp4']:
                    count = (model_df['predicted_original'] == exp).sum()
                    dist.append(count / len(model_df) * 100)
                model_data[model] = dist
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            display_labels = ['Correct\n(Metaphorical)', 'Distractor 1\n(Concrete)', 
                            'Distractor 2\n(Abstract)', 'Distractor 3\n(Antonym)']
            x = np.arange(len(display_labels))
            width = 0.8 / len(models)
            
            colors_palette = plt.cm.Set2(np.linspace(0, 1, len(models)))
            
            for i, model in enumerate(models):
                offset = (i - len(models)/2 + 0.5) * width
                bars = ax.bar(x + offset, model_data[model], width, 
                             label=model, color=colors_palette[i], alpha=0.8, 
                             edgecolor='black', linewidth=0.5)
                
                # Add value labels on bars for correct answers only
                if model_data[model][0] > 20:  # Only label if significant
                    ax.text(offset, model_data[model][0] + 1,
                           f'{model_data[model][0]:.0f}%',
                           ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Response Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage of Predictions (%)', fontsize=12, fontweight='bold')
            
            prompt_name = 'Prompt V1' if prompt_type == 'met_v1' else 'Prompt V2'
            ax.set_title(f'Figure {prompt_idx}: Distractor Analysis per Model - {prompt_name}',
                        fontsize=14, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels(display_labels)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(output_path / f'figure{prompt_idx}_distractor_per_model_{prompt_type}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def _create_figure2_type_analysis(self, output_path):
        """Figure 2: Error distribution by type (averaged over prompts)"""
        
        # Add type information to dataframe using per-dataset logic based on
        # the original dataset identifier `source_dataset_short`.
        df_with_type = self.df.copy()

        if not self.dataset_metadata:
            print("No dataset metadata available for type analysis")
            return

        def _lookup_type(row):
            ds_key = row.get('source_dataset_short') or row.get('dataset')
            if ds_key is None:
                return None
            return self.dataset_metadata.get(ds_key, {}).get('type_mapping', {}).get(row['idx'], None)

        df_with_type['type'] = df_with_type.apply(_lookup_type, axis=1)
        df_with_type = df_with_type.dropna(subset=['type'])
        df_with_type['type'] = df_with_type['type'].astype(str)

        # Keep only the three valid metaphor types; drop any leftover
        # negative/invalid codes (e.g. -1, -2, -3).
        df_with_type = df_with_type[df_with_type['type'].isin(['1', '2', '3'])]
        
        # Map types to descriptive labels
        type_labels = {
            '1': 'Type 1: Lexicalized',
            '2': 'Type 2: Implicit',
            '3': 'Type 3: Ad-hoc'
        }
        df_with_type['type_label'] = df_with_type['type'].map(lambda x: type_labels.get(x, x))
        
        # Get models
        models = sorted(df_with_type['model'].unique())
        model_display_names = {model: model.split('/')[-1] for model in models}
        
        types = sorted(df_with_type['type_label'].unique())
        
        # Calculate accuracy by model and type
        accuracy_data = {}
        for model in models:
            model_accs = []
            for type_label in types:
                subset = df_with_type[(df_with_type['model'] == model) & 
                                      (df_with_type['type_label'] == type_label)]
                if len(subset) > 0:
                    acc = subset['is_correct'].mean() * 100
                else:
                    acc = 0
                model_accs.append(acc)
            accuracy_data[model] = model_accs

        # # Print a textual summary for Figure 2
        # print("\n" + "="*80)
        # print("Figure 2: Model Performance by Type (Averaged over Prompt Versions)")
        # print("="*80)
        # header = ["Model"] + list(types)
        # print(" | ".join(f"{h:>20}" for h in header))
        # print("-"*80)
        # for model in models:
        #     row_vals = [model_display_names[model]] + [f"{acc:.1f}%" for acc in accuracy_data[model]]
        #     print(" | ".join(f"{v:>20}" for v in row_vals))
        # print("="*80)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = np.arange(len(models)) * 1.2
        width = 0.35
        
        colors = ['#649BD5', '#6FA054', '#E36D3F']  # Blue, Green, Orange

        for i, type_label in enumerate(types):
            values = [accuracy_data[model][i] for model in models]
            bars = ax.bar(x + i * width, values, width, label=type_label,
                          color=colors[i % len(colors)], alpha=0.8)  # , edgecolor='black')

            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                            f'{val:.0f}%', ha='center', va='bottom', fontsize=14, rotation=90)

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        # ax.set_title('Model Performance by Type (Averaged over Prompt Versions)',
        #            fontsize=14, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([model_display_names[m] for m in models], rotation=30, ha='right', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 110)

        plt.tight_layout()
        plt.savefig(output_path / 'figure3_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_figure3_topic_analysis(self, output_path):
        """Figure 3: Error distribution by topic (averaged over prompts)"""
        
        # Use topic information from dataset_metadata (SN_DDO topics derived
        # from the combined v4 dataset) and link it to the results by idx.

        if 'source_dataset_short' not in self.df.columns:
            print("No 'source_dataset_short' column available for topic analysis")
            return

        if 'SN_DDO' not in self.dataset_metadata or \
           'topic_mapping' not in self.dataset_metadata['SN_DDO']:
            print("No topic metadata available for SN_DDO topic analysis")
            return

        # Filter for SN_DDO items only
        ddo_df = self.df[self.df['source_dataset_short'] == 'SN_DDO'].copy()

        if len(ddo_df) == 0:
            print("No SN_DDO data available for topic analysis")
            return

        topic_mapping = self.dataset_metadata['SN_DDO']['topic_mapping']
        ddo_df['topic'] = ddo_df['idx'].map(topic_mapping)
        ddo_df = ddo_df.dropna(subset=['topic'])

        if len(ddo_df) == 0:
            print("No topics available for topic analysis after mapping")
            return

        # Get top topics by frequency
        topic_counts = ddo_df.groupby('topic')['idx'].nunique()
        top_topics = [t for t in topic_counts.nlargest(11).index.tolist()
                      if t not in ['communication', 'psychology']]

        # Calculate accuracy by topic
        topic_accuracy = {}
        for topic in top_topics:
            topic_df = ddo_df[ddo_df['topic'] == topic]
            acc = topic_df['is_correct'].mean() * 100
            count = topic_df['idx'].nunique()
            topic_accuracy[topic] = (acc, count)

        # Sort by accuracy
        sorted_topics = sorted(topic_accuracy.items(), key=lambda x: x[1][0], reverse=True)

        topics = [t[0] for t in sorted_topics]
        accuracies = [t[1][0] for t in sorted_topics]
        counts = [t[1][1] for t in sorted_topics]

        # Print a textual summary for Figure 3
        # print("\n" + "="*80)
        # print("Figure 3: Performance by Source Domain in SN_DDO Dataset")
        # print("(Averaged over Prompt Versions)")
        # print("="*80)
        # print(f"{'Topic':<30} {'Accuracy %':>12} {'n_items':>10}")
        # print("-"*80)
        # for topic, acc, n in zip(topics, accuracies, counts):
        #     print(f"{topic:<30} {acc:>12.1f} {n:>10}")
        # overall_acc = ddo_df['is_correct'].mean() * 100
        # print("-"*80)
        # print(f"{'Overall SN_DDO':<30} {overall_acc:>12.1f}")
        # print("="*80)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color gradient based on accuracy
        colors = plt.cm.Set3(np.linspace(0.2, 1, len(topics)))

        bars = ax.bar(topics, accuracies, color=colors, alpha=1)  # , edgecolor='black', linewidth=1.5)

        # Add value and count labels
        for bar, acc, count in zip(bars, accuracies, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                    f'{acc:.0f}%\n(n={count})', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        # Add overall average line
        overall_acc = ddo_df['is_correct'].mean() * 100
        ax.axhline(y=overall_acc, color='red', linestyle='--', alpha=0.7,
                   label=f'Overall DDO: {overall_acc:.1f}%', linewidth=2)

        ax.set_xlabel('Source Domain', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        # ax.set_title('Performance by Source Domain in DDO Dataset (Averaged over Prompt Versions)',
        #            fontsize=14, fontweight='bold')
        ax.set_xticklabels(topics, rotation=30, ha='right', fontsize=14)
        ax.legend(fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(output_path / 'figure4_domain_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def print_summary(self):
        """Print concise summary statistics"""
        print("=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total evaluations: {len(self.df)}")
        print(f"Overall accuracy: {self.df['is_correct'].mean():.2%}")
        
        for prompt_type in ['met_v1', 'met_v2']:
            prompt_df = self.df[self.df['prompt_type'] == prompt_type]
            prompt_name = 'Prompt V1' if prompt_type == 'met_v1' else 'Prompt V2'
            print(f"\n{prompt_name}:")
            print(f"  Accuracy: {prompt_df['is_correct'].mean():.2%}")
            if 'chose_dont_know' in prompt_df.columns:
                print(f"  Don't know rate: {prompt_df['chose_dont_know'].mean():.2%}")

    def _save_analysis_summary(self, output_path: Path):
        """Compute and save a JSON summary of key analyses, and print per-dataset stats."""

        summary = {}

        # Overall and per-prompt accuracy / don't-know
        overall_acc = float(self.df['is_correct'].mean()) if 'is_correct' in self.df.columns else None
        summary['overall'] = {
            'total_evaluations': int(len(self.df)),
            'accuracy': overall_acc,
        }

        per_prompt = {}
        for prompt_type in sorted(self.df['prompt_type'].dropna().unique()):
            prompt_df = self.df[self.df['prompt_type'] == prompt_type]
            acc = float(prompt_df['is_correct'].mean()) if len(prompt_df) > 0 else None
            dk = float(prompt_df['chose_dont_know'].mean()) if 'chose_dont_know' in prompt_df.columns and len(prompt_df) > 0 else None
            per_prompt[prompt_type] = {
                'n': int(len(prompt_df)),
                'accuracy': acc,
                'dont_know_rate': dk,
            }
        summary['per_prompt'] = per_prompt

        # Per-model, per-prompt accuracy / don't-know (similar to Table 2)
        per_model_prompt = {}
        for model in sorted(self.df['model'].dropna().unique()):
            model_data = {}
            model_df = self.df[self.df['model'] == model]
            for prompt_type in sorted(model_df['prompt_type'].dropna().unique()):
                mp_df = model_df[model_df['prompt_type'] == prompt_type]
                acc = float(mp_df['is_correct'].mean()) if len(mp_df) > 0 else None
                dk = float(mp_df['chose_dont_know'].mean()) if 'chose_dont_know' in mp_df.columns and len(mp_df) > 0 else None
                model_data[prompt_type] = {
                    'n': int(len(mp_df)),
                    'accuracy': acc,
                    'dont_know_rate': dk,
                }
            per_model_prompt[model] = model_data
        summary['per_model_prompt'] = per_model_prompt

        # Per-dataset accuracy and don't-know, preferring source_dataset_short
        dataset_col = None
        for candidate in ['source_dataset_short', 'source_dataset', 'dataset']:
            if candidate in self.df.columns:
                dataset_col = candidate
                break

        per_dataset = {}
        if dataset_col is not None:
            for ds in sorted(self.df[dataset_col].dropna().unique()):
                ds_df = self.df[self.df[dataset_col] == ds]
                acc = float(ds_df['is_correct'].mean()) if len(ds_df) > 0 else None
                dk = float(ds_df['chose_dont_know'].mean()) if 'chose_dont_know' in ds_df.columns and len(ds_df) > 0 else None
                per_dataset[ds] = {
                    'n': int(len(ds_df)),
                    'accuracy': acc,
                    'dont_know_rate': dk,
                }
        summary['per_dataset'] = per_dataset

        # Type-level summary (mirrors Figure 2 logic)
        type_summary = {}
        if self.dataset_metadata:
            df_with_type = self.df.copy()

            def _lookup_type(row):
                ds_key = row.get('source_dataset_short') or row.get('dataset')
                if ds_key is None:
                    return None
                return self.dataset_metadata.get(ds_key, {}).get('type_mapping', {}).get(row['idx'], None)

            df_with_type['type'] = df_with_type.apply(_lookup_type, axis=1)
            df_with_type = df_with_type.dropna(subset=['type'])
            df_with_type['type'] = df_with_type['type'].astype(str)
            df_with_type = df_with_type[df_with_type['type'].isin(['1', '2', '3'])]

            for model in sorted(df_with_type['model'].dropna().unique()):
                m_df = df_with_type[df_with_type['model'] == model]
                model_types = {}
                for t in sorted(m_df['type'].unique()):
                    t_df = m_df[m_df['type'] == t]
                    acc = float(t_df['is_correct'].mean()) if len(t_df) > 0 else None
                    model_types[t] = {
                        'n': int(len(t_df)),
                        'accuracy': acc,
                    }
                type_summary[model] = model_types
        summary['per_model_type'] = type_summary

        # Topic-level summary for SN_DDO (mirrors Figure 3 logic)
        topic_summary = {}
        if 'source_dataset_short' in self.df.columns and \
           'SN_DDO' in self.dataset_metadata and \
           'topic_mapping' in self.dataset_metadata['SN_DDO']:
            ddo_df = self.df[self.df['source_dataset_short'] == 'SN_DDO'].copy()
            if len(ddo_df) > 0:
                topic_mapping = self.dataset_metadata['SN_DDO']['topic_mapping']
                ddo_df['topic'] = ddo_df['idx'].map(topic_mapping)
                ddo_df = ddo_df.dropna(subset=['topic'])
                if len(ddo_df) > 0:
                    topic_counts = ddo_df.groupby('topic')['idx'].nunique()
                    top_topics = [t for t in topic_counts.nlargest(10).index.tolist() 
                                  if t not in ['communication', 'psychology']]
                    for topic in top_topics:
                        t_df = ddo_df[ddo_df['topic'] == topic]
                        acc = float(t_df['is_correct'].mean()) if len(t_df) > 0 else None
                        n_items = int(t_df['idx'].nunique())
                        topic_summary[topic] = {
                            'n_items': n_items,
                            'accuracy': acc,
                        }
                    overall_acc_sn_ddo = float(ddo_df['is_correct'].mean()) if len(ddo_df) > 0 else None
                    topic_summary['_overall_SN_DDO'] = {
                        'accuracy': overall_acc_sn_ddo,
                        'n_items': int(ddo_df['idx'].nunique()),
                    }
        summary['topics_SN_DDO'] = topic_summary

        # Save summary JSON
        summary_path = output_path / 'analysis_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Analysis summary saved to: {summary_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze LLM evaluation errors')
    parser.add_argument('results_file', help='Path to results JSON file')
    parser.add_argument('--config', '-c', help='Path to config YAML file for metadata analysis')
    parser.add_argument('--plot-dir', default='plots', help='Directory for plot output')
    
    args = parser.parse_args()
    
    analyzer = ErrorAnalyzer(args.results_file, args.config)
    analyzer.print_summary()
    analyzer.create_all_figures(args.plot_dir)

if __name__ == "__main__":
    main()
