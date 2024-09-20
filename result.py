import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=120)
args = parser.parse_args()

pattern = re.compile(f'(proposed|speed|random)_{args.n_clients}_[0-9.]+\.csv')
csv_files = [file for file in os.listdir('.') if pattern.match(file)]

test_accuracy_data = [np.loadtxt(file, delimiter=',') for file in csv_files]
test_accuracy_data = np.vstack(test_accuracy_data)

rounds = np.arange(1, 301)

plt.plot(rounds, test_accuracy_data[0, :], label='proposed')
plt.plot(rounds, test_accuracy_data[1, :], label='speed')
plt.plot(rounds, test_accuracy_data[2, :], label='random')

plt.xlabel('Rounds')
plt.ylabel('Test Accuracy')
plt.title(f'Test Accuracy over Rounds (n_clients: {args.n_clients})')
plt.grid(True)

plt.legend()
plt.savefig(f'testacc_{args.n_clients}.png')