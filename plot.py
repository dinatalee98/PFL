import matplotlib.pyplot as plt
import pandas as pd
import os
from arguments import args_parser


ALGORITHMS = {
    "PUFL": "proposed",
    "PT": "pipeline",
    "UBS": "utility",
    "FedAvg": "random"
}

def inspect_target_accuracy(algorithm_name: str, ma_data: list, target_accuracy: float) -> None:
    for i, acc in enumerate(ma_data):
        if acc >= target_accuracy:
            ma_accuracy = acc
            round_number = i + 1
            print(f"{algorithm_name}: First round achieving target accuracy is Round {round_number} (MA Accuracy: {ma_accuracy:.3f})")
            break
    else:
        print(f"{algorithm_name}: Target accuracy not achieved within the recorded rounds")


def main() -> None:
    args = args_parser()
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 26
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['legend.fontsize'] = 20
    
    df = pd.read_csv(f"./{args.source_folder}/{args.dataset}_{args.n_clients}_{args.subchannels}.csv")
    
    if args.max_round is not None:
        df = df[df['round'] <= args.max_round]
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

    for alg_name in ALGORITHMS.keys():
        ax.plot(df['round'], df[ALGORITHMS[alg_name]], label=alg_name, linewidth=2.0)
        
        if args.target_accuracy is not None:
            inspect_target_accuracy(alg_name, df[alg_name].tolist(), args.target_accuracy)

    ax.set_xlabel("Round", labelpad=15)
    ax.set_ylabel("Test Accuracy (%)", labelpad=15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc='lower right', frameon=False, ncol=2)
    
    ax.tick_params(axis='both', which='major', direction='in', length=6)
    ax.tick_params(axis='both', which='minor', direction='in', length=3)

    if not df.empty:
        ax.set_xlim(0, df['round'].max())

    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.12)
    fig.savefig(f"./{args.result_path}/{args.dataset}_{args.n_clients}_{args.subchannels}.pdf", bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
