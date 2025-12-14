import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path

# Remove Chinese font settings since we're using English
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

def extract_sparsity_from_filename(filename):
    """Extract sparsity ratio from filename"""
    # Match x in sdpa-x.json
    match = re.search(r'sdpa-(\d+\.?\d*)\.json', filename)
    if match:
        return float(match.group(1))
    return None

def load_accuracy_data(base_path, task_name):
    """Load accuracy data for specified task"""
    data = {}
    
    # Find all matching files
    pattern = f"evaluation_{task_name}-5-qwen2-7B-gqa-01full-sdpa-*.json"
    files = list(Path(base_path).glob(pattern))
    
    for file_path in files:
        sparsity = extract_sparsity_from_filename(file_path.name)
        if sparsity is not None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Find the value after "acc,none":
                    match = re.search(r'"acc,none":\s*([\d.]+)', content)
                    if match:
                        accuracy = float(match.group(1))
                        data[sparsity] = accuracy
                        print(f"Task: {task_name}, Sparsity: {sparsity}, Accuracy: {accuracy}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    return data

def plot_accuracy_vs_sparsity(rte_data, piqa_data, save_path="accuracy_vs_sparsity.png"):
    """Plot accuracy vs sparsity ratio"""
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    sparsity_values = sorted(rte_data.keys())
    rte_accuracies = [rte_data[s] for s in sparsity_values]
    piqa_accuracies = [piqa_data.get(s, 0) for s in sparsity_values]
    
    # Plot lines
    plt.plot(sparsity_values, rte_accuracies, 'o-', linewidth=2, markersize=8, 
             label='RTE', color='blue', alpha=0.8)
    plt.plot(sparsity_values, piqa_accuracies, 's-', linewidth=2, markersize=8, 
             label='PIQA', color='red', alpha=0.8)
    
    # Set chart properties
    plt.xlabel('Sparsity Ratio', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Model Accuracy Comparison at Different Sparsity Ratios\nRTE vs PIQA Tasks', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set x-axis ticks
    plt.xticks(sparsity_values, [f'{x:.1f}' for x in sparsity_values])
    
    # Add value labels
    for i, (x, y1, y2) in enumerate(zip(sparsity_values, rte_accuracies, piqa_accuracies)):
        plt.annotate(f'{y1:.3f}', (x, y1), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='blue')
        plt.annotate(f'{y2:.3f}', (x, y2), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=9, color='red')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save chart
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {save_path}")
    
    # Show chart
    plt.show()

def main():
    base_path = "/workspace/SparseCache/accuracy/lm_eval"
    
    print("Loading RTE task data...")
    rte_data = load_accuracy_data(base_path, "rte")
    
    print("\nLoading PIQA task data...")
    piqa_data = load_accuracy_data(base_path, "piqa")
    
    print(f"\nRTE data: {rte_data}")
    print(f"PIQA data: {piqa_data}")
    
    if rte_data and piqa_data:
        plot_accuracy_vs_sparsity(rte_data, piqa_data)
    else:
        print("Insufficient data to plot chart")

if __name__ == "__main__":
    main()