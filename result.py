import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=120)
args = parser.parse_args()


def get_first_matching_file(pattern):
    regex = re.compile(pattern)
    for file in os.listdir('.'):
        if regex.match(file):
            return file
    return None

proposed_result = re.compile(f'proposed_{args.n_clients}_[0-9.]+\.csv')
speed_result = re.compile(f'speed_{args.n_clients}_[0-9.]+\.csv')
random_result = re.compile(f'random_{args.n_clients}_[0-9.]+\.csv')

csv_files = [
    get_first_matching_file(proposed_result),
    get_first_matching_file(speed_result),
    get_first_matching_file(random_result),
]

test_accuracy_data = [np.loadtxt(file, delimiter=',') for file in csv_files]
test_accuracy_data = np.vstack(test_accuracy_data)

rounds = np.arange(1, 201)

fon = 13
plt.rcParams['font.family'] = 'Times New Roman'

plt.plot(rounds, test_accuracy_data[0, :200], label='Proposed')
plt.plot(rounds, test_accuracy_data[1, :200], label='PFL')
plt.plot(rounds, test_accuracy_data[2, :200], label='FedAvg')

plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel('Rounds',fontsize = fon)
plt.ylabel('Test Accuracy', fontsize = fon)
plt.grid(True)

plt.legend(fontsize = fon - 1)
plt.tight_layout()

plt.savefig(f'testacc_{args.n_clients}.png', bbox_inches='tight')