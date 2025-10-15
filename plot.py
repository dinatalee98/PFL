import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from arguments import args_parser


def read_metrics(path: str, max_round: int = None) -> Tuple[List[int], List[float], List[float]]:
    rounds: List[int] = []
    accs: List[float] = []
    losses: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True, restkey="_rest")
        for row in reader:
            round_num = int(row["round"])
            if max_round is None or round_num <= max_round:
                rounds.append(round_num)
                accs.append(float(row["test_acc"]))
                losses.append(float(row["test_loss"]))
    return rounds, accs, losses


def moving_average(values: List[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.array(values, dtype=float)
    arr = np.array(values, dtype=float)
    n = arr.shape[0]
    result = np.empty(n, dtype=float)
    cumsum = np.cumsum(arr)
    for i in range(n):
        start = i - window + 1
        if start <= 0:
            total = cumsum[i]
            count = i + 1
        else:
            total = cumsum[i] - cumsum[start - 1]
            count = window
        result[i] = total / float(count)
    return result


def calculate_enhancements(alg_data: dict) -> None:
    print("\n=== Proposed Algorithm Enhancement Analysis ===")
    proposed_accs = alg_data["proposed"]
    
    for alg_name, accs in alg_data.items():
        if alg_name == "proposed":
            continue
            
        min_len = min(len(proposed_accs), len(accs))
        enhancements = []
        
        for i in range(min_len):
            if accs[i] > 0:
                enhancement = ((proposed_accs[i] - accs[i]) / accs[i]) * 100
                enhancements.append(enhancement)
        
        if enhancements:
            avg_enhancement = np.mean(enhancements)
            print(f"Proposed vs {alg_name.capitalize()}: {avg_enhancement:.2f}%")


def inspect_target_accuracy(algorithm_name: str, ma_data: list, target_accuracy: float) -> None:
    for i, acc in enumerate(ma_data):
        if acc >= target_accuracy:
            ma_accuracy = acc
            round_number = i + 1
            print(f"{algorithm_name}: First round achieving target accuracy is Round {round_number} (MA Accuracy: {ma_accuracy:.3f})")
            break
    else:
        print(f"{algorithm_name}: Target accuracy not achieved within the recorded rounds")


def plot_ma(rounds: List[int], accs: List[float], window: int, out_path: str) -> None:
    acc_ma = moving_average(accs, window)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(rounds, acc_ma, label=f"Accuracy MA{window}")
    ax.set_xlabel("Round", fontsize=14)
    ax.set_ylabel("Test Accuracy", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc='lower right')
    
    ax.set_xlim(min(rounds), max(rounds))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = args_parser()

    if args.algorithm == "all":
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        
        algs = ["proposed", "pipeline", "utility", "random"]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        all_rounds = []
        alg_data = {}
        
        for alg in algs:
            file = f"./{args.result_path}/{args.dataset}_{alg}_{args.n_clients}_{args.beta}_{args.subchannels}_{args.lr}.txt"
            rounds, accs, _ = read_metrics(file, args.max_round)
            all_rounds.extend(rounds)
            acc_ma = moving_average(accs, args.window)
            ax.plot(rounds, acc_ma, label=alg.capitalize())
            
            # Store raw accuracy data for enhancement calculation
            alg_data[alg] = accs
            
            # Inspect target accuracy with moving averaged data if provided
            if args.target_accuracy is not None:
                inspect_target_accuracy(alg, acc_ma, args.target_accuracy)
        
        ax.set_xlabel("Round", fontsize=14)
        ax.set_ylabel("Test Accuracy", fontsize=14)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc='lower right')
        
        if all_rounds:
            ax.set_xlim(min(all_rounds), max(all_rounds))
        
        fig.tight_layout()
        out_path = f"./{args.result_path}/{args.dataset}_all_{args.n_clients}_{args.beta}_{args.subchannels}_{args.lr}_ma{args.window}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Calculate and print enhancements
        calculate_enhancements(alg_data)
    else:
        file = f"./{args.result_path}/{args.dataset}_{args.algorithm}_{args.n_clients}_{args.beta}_{args.subchannels}_{args.lr}.txt"
        rounds, accs, _ = read_metrics(file, args.max_round)
        base, _ = os.path.splitext(file)
        out_path = base + f"_ma{args.window}.png"
        plot_ma(rounds, accs, args.window, out_path)
        
        if args.target_accuracy is not None:
            acc_ma = moving_average(accs, args.window)
            inspect_target_accuracy(args.algorithm, acc_ma, args.target_accuracy)


if __name__ == "__main__":
    main()
