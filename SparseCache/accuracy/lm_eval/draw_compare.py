import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path

def extract_sparsity_from_filename(filename):
    """Extract sparsity ratio from filename"""
    # Match x in sdpa-x.json
    match = re.search(r'sdpa-(\d+\.?\d*)\.json', filename)
    if match:
        return float(match.group(1))
    return None

def load_accuracy_data(base_path, method_name):
    """Load accuracy data for specified method"""
    data = {}
    
    # Find all matching files
    pattern = f"evaluation_rte-5-qwen2-7B-{method_name}-sdpa-*.json"
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
                        print(f"Method: {method_name}, Sparsity: {sparsity}, Accuracy: {accuracy}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    return data

def plot_method_comparison(topk_data, gqa_data, save_path="method_comparison.png"):
    """Plot accuracy comparison between topk and gqa-01full methods"""
    plt.figure(figsize=(12, 8))
    
    # Get common sparsity values
    common_sparsities = sorted(set(topk_data.keys()) & set(gqa_data.keys()))
    
    if not common_sparsities:
        print("No common sparsity values found between the two methods")
        return
    
    topk_accuracies = [topk_data[s] for s in common_sparsities]
    gqa_accuracies = [gqa_data[s] for s in common_sparsities]
    
    # Plot lines
    plt.plot(common_sparsities, topk_accuracies, 'o-', linewidth=2, markersize=8, 
             label='TopK', color='blue', alpha=0.8)
    plt.plot(common_sparsities, gqa_accuracies, 's-', linewidth=2, markersize=8, 
             label='cache-dynamic_k', color='red', alpha=0.8)
    
    # Set chart properties
    plt.xlabel('Sparsity Ratio', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('RTE Task: Accuracy Comparison Between TopK and GQA-01Full Methods', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set x-axis ticks
    plt.xticks(common_sparsities, [f'{x:.1f}' for x in common_sparsities])
    
    # Add value labels
    for i, (x, y1, y2) in enumerate(zip(common_sparsities, topk_accuracies, gqa_accuracies)):
        plt.annotate(f'{y1:.3f}', (x, y1), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='blue')
        plt.annotate(f'{y2:.3f}', (x, y2), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=9, color='red')
    
    # Add difference annotations
    # for i, (x, y1, y2) in enumerate(zip(common_sparsities, topk_accuracies, gqa_accuracies)):
    #     diff = y1 - y2
    #     color = 'green' if diff > 0 else 'orange'
    #     plt.annotate(f'Δ{diff:+.3f}', (x, max(y1, y2)), textcoords="offset points", 
    #                 xytext=(0,25), ha='center', fontsize=8, color=color, 
    #                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Set y-axis limits for better visualization
    all_accuracies = topk_accuracies + gqa_accuracies
    y_min = min(all_accuracies) - 0.02
    y_max = max(all_accuracies) + 0.05
    plt.ylim(y_min, y_max)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save chart
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {save_path}")
    
    # Show chart
    plt.show()
    
    # Print comparison summary
    print("\n=== Comparison Summary ===")
    for sparsity in common_sparsities:
        topk_acc = topk_data[sparsity]
        gqa_acc = gqa_data[sparsity]
        diff = topk_acc - gqa_acc
        better = "TopK" if diff > 0 else "GQA-01Full"
        print(f"Sparsity {sparsity:.1f}: TopK={topk_acc:.3f}, GQA-01Full={gqa_acc:.3f}, "
              f"Difference={diff:+.3f} ({better} is better)")

def main():
    base_path = "/workspace/SparseCache/accuracy/lm_eval"
    
    print("Loading TopK method data...")
    topk_data = load_accuracy_data(base_path, "topk")
    
    print("\nLoading GQA-01Full method data...")
    gqa_data = load_accuracy_data(base_path, "gqacahe-dynamic_k")
    
    print(f"\nTopK data: {topk_data}")
    print(f"GQA-01Full data: {gqa_data}")
    
    if topk_data and gqa_data:
        plot_method_comparison(topk_data, gqa_data)
    else:
        print("Insufficient data to plot comparison chart")

if __name__ == "__main__":
    main()