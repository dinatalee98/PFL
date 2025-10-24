import os
import pandas as pd
import glob
from arguments import args_parser

def calculate_moving_average(data_list, window_size=50):
    if len(data_list) < window_size:
        return data_list
    
    moving_avg = []
    for i in range(len(data_list)):
        start_idx = max(0, i - window_size + 1)
        window_data = data_list[start_idx:i+1]
        moving_avg.append(sum(window_data) / len(window_data))
    
    return moving_avg

def average_seed_results(all_seed_accuracies):
    if not all_seed_accuracies:
        return []
    
    min_length = min(len(acc) for acc in all_seed_accuracies)
    averaged_accuracies = []
    
    for i in range(min_length):
        avg_acc = sum(acc[i] for acc in all_seed_accuracies) / len(all_seed_accuracies)
        averaged_accuracies.append(avg_acc)
    
    return averaged_accuracies

def integrate_training_results():
    args = args_parser()
    datasets = ['mnist', 'fashion-mnist']
    client_numbers = [50, 100]
    subchannels = [2, 4]
    seeds = [1, 2, 3, 4, 5]
    max_rounds = args.max_round
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    for dataset in datasets:
        for client_num in client_numbers:
            for subchannel in subchannels:
                csv_filename = f"{args.result_path}/{dataset}_{client_num}_{subchannel}.csv"
                
                algorithms = ['proposed', 'pipeline', 'utility', 'random']
                
                all_data = {}
                
                for algorithm in algorithms:
                    all_seed_accuracies = []
                    
                    for seed in seeds:
                        pattern = f"{args.source_folder}/{dataset}_{algorithm}_{client_num}_0.05_{subchannel}_0.01_{seed}.txt"
                        files = glob.glob(pattern)
                        
                        if files:
                            file_path = files[0]
                            try:
                                with open(file_path, 'r') as f:
                                    lines = f.readlines()
                                
                                accuracies = []
                                
                                for line in lines[1:max_rounds+1 if max_rounds else len(lines)]:
                                    parts = [part.strip() for part in line.strip().split(',')]
                                    if len(parts) >= 2:
                                        accuracies.append(float(parts[1]))
                                
                                if accuracies:
                                    all_seed_accuracies.append(accuracies)
                                
                            except Exception as e:
                                print(f"Error reading {file_path}: {e}")
                    
                    if all_seed_accuracies:
                        averaged_accuracies = average_seed_results(all_seed_accuracies)
                        moving_avg_accuracies = calculate_moving_average(averaged_accuracies)
                        all_data[algorithm] = moving_avg_accuracies
                
                if all_data:
                    df = pd.DataFrame(all_data)
                    df.index = range(1, len(df) + 1)
                    df.index.name = 'round'
                    df.to_csv(csv_filename)
                    print(f"Created {csv_filename} with {len(df)} rounds")

if __name__ == "__main__":
    integrate_training_results()
