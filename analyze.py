import pandas as pd
import numpy as np
import argparse
import os
import glob
from arguments import args_parser


def analyze_results(source_folder, dataset, n_clients, subchannels, target_accuracy=None):
    """
    Analyze federated learning results for target accuracy achievement and model comparison.
    
    Args:
        source_folder (str): Path to folder containing CSV result files
        dataset (str): Dataset name (e.g., 'mnist', 'fashion-mnist')
        n_clients (int): Number of clients
        subchannels (int): Number of subchannels
        target_accuracy (float): Target accuracy threshold to analyze
    """
    
    # Construct the expected filename pattern
    expected_filename = f"{dataset}_{n_clients}_{subchannels}.csv"
    csv_file = os.path.join(source_folder, expected_filename)
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{expected_filename}' not found in {source_folder}")
        return
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get model columns (exclude 'round' column)
        model_columns = [col for col in df.columns if col != 'round']
        
        if not model_columns:
            print("No model columns found in the CSV file")
            return
        
        # Check if 'proposed' algorithm exists
        if 'proposed' not in model_columns:
            print("Error: 'proposed' algorithm not found in the data")
            return
        
        print(f"Analysis: {dataset} | Clients: {n_clients} | Subchannels: {subchannels}")
        print("=" * 60)
        
        # 1. Target Accuracy Rounds
        if target_accuracy is not None:
            print(f"\n1. ROUNDS TO REACH {target_accuracy}% ACCURACY")
            print("-" * 50)
            
            # Find rounds for each algorithm
            rounds_to_target = {}
            for model in model_columns:
                target_reached_mask = df[model] >= target_accuracy
                if target_reached_mask.any():
                    first_target_round = df[target_reached_mask]['round'].iloc[0]
                    rounds_to_target[model] = first_target_round
                else:
                    rounds_to_target[model] = None
            
            # Display rounds and gaps
            proposed_round = rounds_to_target.get('proposed')
            print(f"{'Algorithm':<12s} {'Round':<8s} {'Gap vs Proposed':<15s}")
            print("-" * 50)
            
            for model in model_columns:
                round_val = rounds_to_target[model]
                if round_val is not None:
                    if model == 'proposed':
                        print(f"{model:<12s} {round_val:<8d} {'baseline':<15s}")
                    else:
                        if proposed_round is not None:
                            gap_percentage = ((round_val - proposed_round) / proposed_round) * 100
                            gap_str = f"{gap_percentage:+.1f}%"
                        else:
                            gap_str = "N/A"
                        print(f"{model:<12s} {round_val:<8d} {gap_str:<15s}")
                else:
                    print(f"{model:<12s} {'N/A':<8s} {'Not reached':<15s}")
        
        # 2. Comprehensive Performance Analysis
        print(f"\n2. COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("-" * 80)
        
        proposed_final = df['proposed'].iloc[-1]
        proposed_avg = df['proposed'].mean()
        
        print(f"{'Algorithm':<12s} {'Final Acc':<12s} {'Avg Acc':<12s} {'Final Gap':<12s} {'Avg Gap':<12s}")
        print("-" * 80)
        
        for model in model_columns:
            final_acc = df[model].iloc[-1]
            avg_acc = df[model].mean()
            
            if model == 'proposed':
                print(f"{model:<12s} {final_acc:>8.2f}% {avg_acc:>8.2f}% {'baseline':<12s} {'baseline':<12s}")
            else:
                final_gap = final_acc - proposed_final
                avg_gap = avg_acc - proposed_avg
                print(f"{model:<12s} {final_acc:>8.2f}% {avg_acc:>8.2f}% {final_gap:>+8.2f}% {avg_gap:>+8.2f}%")
        
    except Exception as e:
        print(f"Error analyzing {csv_file}: {str(e)}")
        return


def main():
    args = args_parser()
    
    if not args.source_folder:
        print("Error: --source_folder argument is required")
        return
    
    if not os.path.exists(args.source_folder):
        print(f"Error: Source folder '{args.source_folder}' does not exist")
        return
    
    analyze_results(args.source_folder, args.dataset, args.n_clients, args.subchannels, args.target_accuracy)


if __name__ == "__main__":
    main()
