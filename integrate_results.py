import glob
import os
from collections import defaultdict

import pandas as pd

from arguments import args_parser


ALGORITHMS = ["proposed", "pipeline", "utility", "random", "client_selection"]
ALGORITHM_ALIASES = {"client_selection": "utility"}


def calculate_moving_average(data_list, window_size):
    if not data_list:
        return []
    moving_avg = []
    for idx in range(len(data_list)):
        start_idx = max(0, idx - window_size + 1)
        window_data = data_list[start_idx : idx + 1]
        moving_avg.append(sum(window_data) / len(window_data))
    return moving_avg


def average_seed_results(all_seed_accuracies):
    if not all_seed_accuracies:
        return []
    min_length = min(len(acc) for acc in all_seed_accuracies)
    return [
        sum(acc[idx] for acc in all_seed_accuracies) / len(all_seed_accuracies)
        for idx in range(min_length)
    ]


def parse_result_filename(file_path, dataset):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    dataset_prefix = f"{dataset}_"
    if not stem.startswith(dataset_prefix):
        return None

    remainder = stem[len(dataset_prefix) :]
    algorithm = None
    suffix = None
    for algo in ALGORITHMS:
        algo_prefix = f"{algo}_"
        if remainder.startswith(algo_prefix):
            algorithm = ALGORITHM_ALIASES.get(algo, algo)
            suffix = remainder[len(algo_prefix) :]
            break

    if algorithm is None or suffix is None:
        return None

    tokens = suffix.split("_")
    if len(tokens) < 3:
        return None

    try:
        n_clients = int(tokens[0])
        seed = int(tokens[-1])
    except ValueError:
        return None

    subchannel = None
    for token in tokens[1:-1]:
        if token.isdigit():
            subchannel = int(token)
            break

    if subchannel is None:
        return None

    return algorithm, n_clients, subchannel, seed


def read_test_accuracies(file_path, max_rounds):
    accuracies = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    data_lines = lines[1:]
    if max_rounds is not None:
        data_lines = data_lines[:max_rounds]

    for line in data_lines:
        parts = [part.strip() for part in line.strip().split(",", 2)]
        if len(parts) < 2:
            continue
        try:
            accuracies.append(float(parts[1]))
        except ValueError:
            continue

    return accuracies


def integrate_training_results():
    args = args_parser()
    dataset = args.dataset
    max_rounds = args.max_round
    smoothing_window = max(1, args.window)

    source_root = args.source_folder if args.source_folder else args.result_path
    source_folder = source_root
    if os.path.isdir(os.path.join(source_root, dataset)):
        source_folder = os.path.join(source_root, dataset)

    output_folder = args.result_path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    txt_files = glob.glob(os.path.join(source_folder, "*.txt"))
    grouped_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file_path in txt_files:
        parsed = parse_result_filename(file_path, dataset)
        if parsed is None:
            continue
        algorithm, n_clients, subchannel, seed = parsed
        grouped_files[(n_clients, subchannel)][algorithm][seed].append(file_path)

    for (n_clients, subchannel), algorithm_map in sorted(grouped_files.items()):
        all_data = {}

        for algorithm, seed_map in algorithm_map.items():
            seed_accuracies = []
            for _, file_paths in sorted(seed_map.items()):
                per_seed_runs = []
                for file_path in sorted(file_paths):
                    try:
                        accuracies = read_test_accuracies(file_path, max_rounds)
                    except Exception as exc:
                        print(f"Error reading {file_path}: {exc}")
                        continue
                    if accuracies:
                        per_seed_runs.append(accuracies)
                if per_seed_runs:
                    seed_accuracies.append(average_seed_results(per_seed_runs))

            if not seed_accuracies:
                continue

            averaged_accuracies = average_seed_results(seed_accuracies)
            all_data[algorithm] = calculate_moving_average(averaged_accuracies, smoothing_window)

        if not all_data:
            continue

        min_length = min(len(values) for values in all_data.values())
        trimmed_data = {
            algorithm: values[:min_length]
            for algorithm, values in sorted(all_data.items())
        }
        df = pd.DataFrame(trimmed_data)
        df.insert(0, "round", range(1, len(df) + 1))
        csv_filename = os.path.join(output_folder, f"{dataset}_{n_clients}_{subchannel}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Created {csv_filename} with {len(df)} rounds")

if __name__ == "__main__":
    integrate_training_results()
