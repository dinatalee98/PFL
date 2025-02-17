import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=100)
round = 100
args = parser.parse_args()

def get_first_matching_file(pattern):
    regex = re.compile(pattern)
    folder_path = './sample/'
    
    if not os.path.exists(folder_path):
        print(f"Error: Directory {folder_path} does not exist!")
        return None
    
    for file in os.listdir(folder_path):
        if regex.match(file):
            return os.path.join(folder_path, file)
    return None

# Corrected regex patterns

proposed_result = re.compile(f'proposed_{args.n_clients}.csv')
fedavg_result = re.compile(f'fedavg_{args.n_clients}.csv')
pipeline_result = re.compile(f'pipeline_{args.n_clients}.csv')
selection_result = re.compile(f'client_selection_{args.n_clients}.csv')

csv_files = [
    get_first_matching_file(proposed_result),
    get_first_matching_file(fedavg_result),
    get_first_matching_file(pipeline_result),
    get_first_matching_file(selection_result)
]

# Check for missing files before loading data
if None in csv_files:
    print("Error: Some CSV files were not found!")
    print("Files found:", csv_files)
    exit(1)

# Load data
try:
    test_accuracy_data = [np.loadtxt(file, delimiter=',') for file in csv_files]
    test_accuracy_data = np.vstack(test_accuracy_data)
except Exception as e:
    print("Error loading CSV files:", e)
    exit(1)

rounds = np.arange(1, round + 1)
fon = 13
plt.rcParams['font.family'] = 'Times New Roman'

plt.plot(rounds, test_accuracy_data[0, :round], label='Proposed')
plt.plot(rounds, test_accuracy_data[1, :round], label='FedAvg')
plt.plot(rounds, test_accuracy_data[2, :round], label='Pipeline')
plt.plot(rounds, test_accuracy_data[3, :round], label='Selection')

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Rounds', fontsize=fon)
plt.ylabel('Test Accuracy', fontsize=fon)
plt.grid(True)

plt.legend(fontsize=fon - 1)
plt.tight_layout()
plt.savefig(f'./testacc_{args.n_clients}.png', bbox_inches='tight')
