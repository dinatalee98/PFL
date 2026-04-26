import argparse
import glob
import os
from collections import defaultdict

import pandas as pd


ALGORITHM_ALIASES = {
    "client_selection": "utility",
}


def parse_txt_filename(path: str):
    base_name = os.path.basename(path)
    stem, ext = os.path.splitext(base_name)
    if ext != ".txt":
        return None

    parts = stem.split("_")
    if len(parts) < 7:
        return None

    dataset = parts[0]
    first_int_idx = None
    for idx in range(1, len(parts)):
        if parts[idx].isdigit():
            first_int_idx = idx
            break

    if first_int_idx is None or first_int_idx < 2:
        return None

    algorithm = "_".join(parts[1:first_int_idx])

    try:
        n_clients = int(parts[first_int_idx])
        subchannels = int(parts[first_int_idx + 2])
    except ValueError:
        return None
    except IndexError:
        return None

    if len(parts) <= first_int_idx + 3:
        return None

    setting_tokens = tuple(parts[first_int_idx + 1 : -1])
    seed_token = parts[-1]

    if not seed_token.isdigit():
        return None

    seed = int(seed_token)
    canonical_algorithm = ALGORITHM_ALIASES.get(algorithm, algorithm)
    return dataset, canonical_algorithm, n_clients, subchannels, setting_tokens, seed


def read_accuracy_series(path: str):
    accuracies = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except Exception:
        return None

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",", 2)]
        if len(parts) < 2:
            continue
        try:
            accuracies.append(float(parts[1]))
        except ValueError:
            continue
    return accuracies


def average_seed_results(all_seed_accuracies):
    if not all_seed_accuracies:
        return []
    min_length = min(len(acc) for acc in all_seed_accuracies)
    averaged = []
    for idx in range(min_length):
        averaged.append(sum(acc[idx] for acc in all_seed_accuracies) / len(all_seed_accuracies))
    return averaged


def calculate_moving_average(values, window):
    if not values:
        return []
    output = []
    for idx in range(len(values)):
        start_idx = max(0, idx - window + 1)
        window_values = values[start_idx : idx + 1]
        output.append(sum(window_values) / len(window_values))
    return output


def setting_suffix(setting_tokens):
    return "_".join(setting_tokens).replace(".", "p")


def convert_folder(source_folder: str, output_folder: str, window: int):
    txt_files = sorted(glob.glob(os.path.join(source_folder, "*.txt")))
    grouped_files = defaultdict(lambda: defaultdict(list))

    for txt_path in txt_files:
        parsed = parse_txt_filename(txt_path)
        if parsed is None:
            continue
        dataset, algorithm, n_clients, subchannels, setting_tokens, _ = parsed
        grouped_files[(dataset, n_clients, subchannels, setting_tokens)][algorithm].append(txt_path)

    created_files = []
    base_key_counts = defaultdict(int)
    for dataset, n_clients, subchannels, _ in grouped_files.keys():
        base_key_counts[(dataset, n_clients, subchannels)] += 1

    for (dataset, n_clients, subchannels, setting_tokens), algorithm_map in grouped_files.items():
        all_data = {}

        for algorithm, paths in algorithm_map.items():
            seed_accuracies = []
            for txt_path in paths:
                accuracy_list = read_accuracy_series(txt_path)
                if accuracy_list:
                    seed_accuracies.append(accuracy_list)

            if not seed_accuracies:
                continue

            averaged = average_seed_results(seed_accuracies)
            if not averaged:
                continue
            all_data[algorithm] = calculate_moving_average(averaged, window)

        if not all_data:
            continue

        min_length = min(len(values) for values in all_data.values())
        trimmed_data = {algorithm: values[:min_length] for algorithm, values in all_data.items()}
        merged_df = pd.DataFrame(trimmed_data)
        merged_df.index = range(1, len(merged_df) + 1)
        merged_df.index.name = "round"

        base_output_name = f"{dataset}_{n_clients}_{subchannels}"
        if base_key_counts[(dataset, n_clients, subchannels)] > 1:
            output_name = f"{base_output_name}_{setting_suffix(setting_tokens)}.csv"
        else:
            output_name = f"{base_output_name}.csv"
        output_path = os.path.join(output_folder, output_name)
        merged_df.to_csv(output_path)
        created_files.append(output_path)

    return created_files


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_folder", required=True, help="folder containing txt files")
    parser.add_argument("--output_folder", default=None, help="folder to save csv files")
    parser.add_argument("--window", type=int, default=1, help="moving average window size")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    source_folder = args.source_folder
    output_folder = args.output_folder or source_folder
    os.makedirs(output_folder, exist_ok=True)

    if args.window < 1:
        raise ValueError("--window must be at least 1")

    created_files = convert_folder(source_folder, output_folder, args.window)
    if not created_files:
        print("No CSV files were created.")
        return

    for path in created_files:
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
