import os
import pandas as pd
import glob

def integrate_training_results():
    datasets = ['mnist', 'fashion-mnist']
    client_numbers = [50, 100]
    subchannels = [2, 4]
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    for dataset in datasets:
        for client_num in client_numbers:
            for subchannel in subchannels:
                csv_filename = f"results/{dataset}_{client_num}_{subchannel}.csv"
                
                result_folder = f"result_{dataset}"
                algorithms = ['proposed', 'pipeline', 'utility', 'random']
                
                all_data = {}
                
                for algorithm in algorithms:
                    pattern = f"{result_folder}/{dataset}_{algorithm}_{client_num}_0.05_{subchannel}_0.01.txt"
                    files = glob.glob(pattern)
                    
                    if files:
                        file_path = files[0]
                        try:
                            with open(file_path, 'r') as f:
                                lines = f.readlines()
                            
                            rounds = []
                            accuracies = []
                            
                            for line in lines[1:]:
                                parts = [part.strip() for part in line.strip().split(',')]
                                if len(parts) >= 2:
                                    rounds.append(int(parts[0]))
                                    accuracies.append(float(parts[1]))
                            
                            if len(accuracies) >= 50:
                                moving_avg_accuracies = []
                                for i in range(len(accuracies)):
                                    start_idx = max(0, i - 49)
                                    window_data = accuracies[start_idx:i+1]
                                    moving_avg_accuracies.append(sum(window_data) / len(window_data))
                                all_data[algorithm] = moving_avg_accuracies
                            else:
                                all_data[algorithm] = accuracies
                            
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
                
                if all_data:
                    df = pd.DataFrame(all_data)
                    df.index = range(1, len(df) + 1)
                    df.index.name = 'round'
                    df.to_csv(csv_filename)
                    print(f"Created {csv_filename} with {len(df)} rounds")

if __name__ == "__main__":
    integrate_training_results()
