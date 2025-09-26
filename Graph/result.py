import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=40)
round = 300
args = parser.parse_args()

folder = 'alpha01'
def get_first_matching_file(pattern):
    regex = re.compile(pattern)
    folder_path = f'./result/{folder}/'
    if not os.path.exists(folder_path):
        print(f"Error: Directory {folder_path} does not exist!")
        return None
    for file in os.listdir(folder_path):
        if regex.match(file):
            return os.path.join(folder_path, file)
    return None

# Corrected regex patterns
dataset = 'mnist'
proposed_result = re.compile(f'proposed_{args.n_clients}_3_{dataset}.csv')
fedavg_result = re.compile(f'fedavg_{args.n_clients}_3_{dataset}.csv')
fedprox_result = re.compile(f'fedprox_{args.n_clients}_3_{dataset}.csv')
selection_result = re.compile(f'client_selection_{args.n_clients}_3_{dataset}.csv')

csv_files = [
get_first_matching_file(proposed_result),
get_first_matching_file(fedavg_result),
get_first_matching_file(fedprox_result),
get_first_matching_file(selection_result)
]

# Check for missing files before loading data
if None in csv_files:
    print("Error: Some CSV files were not found!")
    print("Files found:", csv_files)
    exit(1)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Load data
try:
    test_accuracy_data = [np.loadtxt(file, delimiter=',') for file in csv_files]
    test_accuracy_data = np.vstack(test_accuracy_data)
except Exception as e:
    print("Error loading CSV files:", e)
    exit(1)

rounds = np.arange(1, test_accuracy_data.shape[1] + 1)
fon = 13
plt.rcParams['font.family'] = 'Times New Roman'

# Define the window size for moving average
window_size = 30

# Plot moving average for each method
plt.plot(rounds[window_size - 1:], moving_average(test_accuracy_data[0, :], window_size), label='Proposed')
plt.plot(rounds[window_size - 1:], moving_average(test_accuracy_data[1, :], window_size), label='FedAvg')
plt.plot(rounds[window_size - 1:], moving_average(test_accuracy_data[2, :], window_size), label='FedProx')
plt.plot(rounds[window_size - 1:], moving_average(test_accuracy_data[3, :], window_size), label='UBS')

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Rounds', fontsize=fon)
plt.ylabel('Test Accuracy (%)', fontsize=fon)
plt.legend(fontsize=fon)
plt.grid(True)
plt.savefig(f'./Graph/{dataset}.png', bbox_inches='tight')