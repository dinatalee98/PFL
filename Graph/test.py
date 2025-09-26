import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=40)
parser.add_argument('--subchannels', type=int, default=3)
round = 300
args = parser.parse_args()

dataset = 'cifar10'

print(os.listdir())
def get_first_matching_file(pattern):
    regex = re.compile(pattern)
    folder_path = './result/beta05/'
    
    if not os.path.exists(folder_path):
        print(f"Error: Directory {folder_path} does not exist!")
        return None
    
    for file in os.listdir(folder_path):
        if regex.match(file):
            return os.path.join(folder_path, file)
    return None

# Corrected regex patterns

proposed_result = re.compile(f'proposed_{args.n_clients}_{args.subchannels}_{dataset}.csv')
fedavg_result = re.compile(f'fedavg_{args.n_clients}_{args.subchannels}_{dataset}.csv')
fedprox_result = re.compile(f'fedprox_{args.n_clients}_{args.subchannels}_{dataset}.csv')
selection_result = re.compile(f'client_selection_{args.n_clients}_{args.subchannels}_{dataset}.csv')
"""
proposed_result = re.compile(f'proposed_{args.n_clients}_{args.subchannels}.csv')
fedavg_result = re.compile(f'fedavg_{args.n_clients}_{args.subchannels}.csv')
pipeline_result = re.compile(f'pipeline_{args.n_clients}_{args.subchannels}.csv')
selection_result = re.compile(f'client_selection_{args.n_clients}_{args.subchannels}.csv')
"""
csv_files = [
    get_first_matching_file(fedavg_result),
    get_first_matching_file(selection_result),
    get_first_matching_file(fedprox_result),
    get_first_matching_file(proposed_result)
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

for i in range(4):
    print(np.average(test_accuracy_data, axis=1)[i])
for i in range(4):
    print(np.std(test_accuracy_data, axis=1)[i])

# Calculate the threshold for the bottom 30%
threshold = np.percentile(test_accuracy_data, 30, axis=1)

# Filter the data to get the bottom 30%
bottom_30_data = [data[data <= thresh] for data, thresh in zip(test_accuracy_data, threshold)]

# Calculate and print the average of the bottom 30% data
bottom_30_avg = [np.mean(data) for data in bottom_30_data]
for i, avg in enumerate(bottom_30_avg):
    print(f"{avg}")