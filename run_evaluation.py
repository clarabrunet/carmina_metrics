import json
import glob
import os
import sys
from pathlib import Path

# Add 'scripts' folder to path so we can import evaluate_model
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from evaluate_model import ModelEvaluator, NormalizationConfig

def main():
    # Setup directories
    input_dir = 'models'
    output_dir = 'results'
    problematic_docs_path = os.path.join('scripts', 'problematic_docs.json')
    
    if not os.path.exists(input_dir):
        print(f"Creating {input_dir} directory - please put your model .json files here!")
        os.makedirs(input_dir)
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Load exclusions
    excluded_docs = set()
    if os.path.exists(problematic_docs_path):
        print(f"Loading excluded documents from {problematic_docs_path}")
        with open(problematic_docs_path, 'r', encoding='utf-8') as f:
            exclusions = json.load(f)
            for item in exclusions:
                if isinstance(item, str):
                    excluded_docs.add(item)
                elif isinstance(item, dict) and 'namefile' in item:
                    excluded_docs.add(item['namefile'])
    else:
        print("Warning: problematic_docs.json not found. Evaluating on full dataset.")

    # Configure Master Evaluator (Best settings by default)
    config = NormalizationConfig(
        normalize_case=True,
        remove_accents=True,
        merge_dates=True,
        standardize_separators=True,
        partial_match=True
    )
    evaluator = ModelEvaluator(config)

    # Find all JSON files
    model_files = glob.glob(os.path.join(input_dir, '*.json'))
    print(f"Found {len(model_files)} model files to evaluate.\n")

    summary_data = []

    for model_path in sorted(model_files):
        filename = os.path.basename(model_path)
        model_name = filename.replace('output_', '').replace('.json', '')
        
        print(f"Processing: {model_name}...")
        
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Run Evaluation
            results = evaluator.evaluate_all(data, excluded_docs=excluded_docs)
            
            # Save Individual Result
            result_output = {
                'model_name': model_name,
                'metrics': results
            }
            result_path = os.path.join(output_dir, f'{model_name}_metrics.json')
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_output, f, indent=2, ensure_ascii=False)
            
            # Add to summary
            summary_data.append({
                'Model': model_name,
                'Docs': results['documents_evaluated'],
                'Identity_Precision': results['identity']['precision'],
                'Identity_Recall': results['identity']['recall'],
                'Identity_F1': results['identity']['f1'],
                'Category_F1': results['category']['f1'],
                'Combined_F1': results['combined']['f1']
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Generate Summary Report
    if summary_data:
        # Save JSON Summary
        summary_path = os.path.join(output_dir, 'models_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Print Table
        print("\n" + "="*95)
        print("FINAL EVALUATION SUMMARY (Clean Dataset)")
        print("="*95)
        print(f"{'Model':<25} {'Docs':<6} {'Id_Prec':<8} {'Id_Rec':<8} {'Id_F1':<8} {'Cat_F1':<8} {'Comb_F1':<8}")
        print("-" * 95)
        
        for row in summary_data:
            print(f"{row['Model']:<25} {row['Docs']:<6} {row['Identity_Precision']:<8.4f} {row['Identity_Recall']:<8.4f} {row['Identity_F1']:<8.4f} {row['Category_F1']:<8.4f} {row['Combined_F1']:<8.4f}")
        
        print("-" * 95)
        print(f"Full results saved in '{output_dir}/'")

if __name__ == '__main__':
    main()
