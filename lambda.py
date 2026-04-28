import ast
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunKey:
    dataset: str
    algorithm: str
    n_clients: int
    beta: float
    subchannels: int
    lr: float
    lambda_stale: float
    tau: float | None
    seed: int


def parse_run_key(filename: str) -> RunKey | None:
    if not filename.endswith(".txt"):
        return None
    parts = filename[:-4].split("_")
    if len(parts) != 9:
        return None
    dataset, algorithm, n_clients, beta, subchannels, lr, lambda_stale, tau, seed = parts
    try:
        parsed_tau = None if tau.lower() in {"none", "null"} else float(tau)
        return RunKey(
            dataset=dataset,
            algorithm=algorithm,
            n_clients=int(n_clients),
            beta=float(beta),
            subchannels=int(subchannels),
            lr=float(lr),
            lambda_stale=float(lambda_stale),
            tau=parsed_tau,
            seed=int(seed),
        )
    except ValueError:
        return None


def read_round_acc(path: str, max_round: int | None) -> tuple[list[int], list[float], Counter]:
    rounds: list[int] = []
    accs: list[float] = []
    participation: Counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return rounds, accs, participation
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",", maxsplit=3)]
            if len(parts) < 2:
                continue
            try:
                r = int(parts[0])
                a = float(parts[1])
            except ValueError:
                continue
            if max_round is not None and r > max_round:
                break
            rounds.append(r)
            accs.append(a)
            if len(parts) >= 4:
                try:
                    clients = ast.literal_eval(parts[3])
                    participation.update(clients)
                except (ValueError, SyntaxError):
                    pass
    return rounds, accs, participation


def jain_fairness(participation: Counter, n_clients: int) -> float:
    counts = [participation.get(k, 0) for k in range(n_clients)]
    s1 = sum(counts)
    s2 = sum(c * c for c in counts)
    if s2 == 0:
        return float("nan")
    return (s1 * s1) / (n_clients * s2)


def moving_average(xs: list[float], window: int) -> list[float]:
    if window is None or window <= 1 or len(xs) < window:
        return xs
    arr = np.asarray(xs, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.convolve(arr, kernel, mode="valid")
    return out.tolist()


def is_selected_lambda(value: float, selected: list[float] | None, atol: float = 1e-12) -> bool:
    if not selected:
        return True
    return any(abs(value - target) <= atol for target in selected)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values))


def main() -> None:
    from arguments import args_parser
    args = args_parser()

    root = getattr(args, "source_folder", None) or getattr(args, "result_path", "result")
    selected_lambdas = getattr(args, "lambda_stale_values", [])
    w = max(1, int(getattr(args, "window", 100) or 100))
    if not os.path.isdir(root):
        raise SystemExit(f"result_path not found: {root}")

    series: dict[float, dict[int, list[tuple[list[int], list[float], RunKey, Counter]]]] = {}
    for name in os.listdir(root):
        key = parse_run_key(name)
        if key is None:
            continue
        if args.dataset is not None and key.dataset != args.dataset:
            continue
        if args.algorithm is not None and key.algorithm != args.algorithm:
            continue
        if args.n_clients is not None and key.n_clients != args.n_clients:
            continue
        if args.beta is not None and key.beta != args.beta:
            continue
        if args.subchannels is not None and key.subchannels != args.subchannels:
            continue
        if args.lr is not None and key.lr != args.lr:
            continue
        if args.tau is not None and key.tau is not None and abs(key.tau - args.tau) > 1e-12:
            continue
        if not is_selected_lambda(key.lambda_stale, selected_lambdas):
            continue

        path = os.path.join(root, name)
        rounds, accs, participation = read_round_acc(path, args.max_round)
        if not rounds:
            continue
        by_lambda = series.setdefault(key.lambda_stale, {})
        by_lambda.setdefault(key.seed, []).append((rounds, accs, key, participation))

    if not series:
        raise SystemExit("no matching result files found")

    keys_sorted = sorted(series.keys())
    print("lambda_stale\tseeds\tAUC (mean±std)\t\tJain_Fairness (mean±std)\tFinal_MA_Acc (mean±std)")
    for lam in keys_sorted:
        seed_groups = series[lam]
        aucs = []
        jfis = []
        final_ma_accs = []
        for seed_entries in seed_groups.values():
            seed_aucs = []
            seed_jfis = []
            seed_final_ma_accs = []
            for rounds, accs, key, participation in seed_entries:
                if len(rounds) >= 2:
                    auc = float(np.trapezoid(accs, rounds) if hasattr(np, "trapezoid") else np.trapz(accs, rounds))
                else:
                    auc = float("nan")
                jfi = jain_fairness(participation, key.n_clients)
                smoothed_accs = moving_average(accs, w)
                final_ma_acc = float(smoothed_accs[-1]) if smoothed_accs else float("nan")
                seed_aucs.append(auc)
                seed_jfis.append(jfi)
                seed_final_ma_accs.append(final_ma_acc)
            seed_mean_auc, _ = mean_std(seed_aucs)
            seed_mean_jfi, _ = mean_std(seed_jfis)
            seed_mean_final_ma_acc, _ = mean_std(seed_final_ma_accs)
            aucs.append(seed_mean_auc)
            jfis.append(seed_mean_jfi)
            final_ma_accs.append(seed_mean_final_ma_acc)
        mean_auc, std_auc = mean_std(aucs)
        mean_jfi, std_jfi = mean_std(jfis)
        mean_final_ma_acc, std_final_ma_acc = mean_std(final_ma_accs)
        print(
            f"{lam:g}\t{len(seed_groups)}\t{mean_auc:.4f}±{std_auc:.4f}\t\t"
            f"{mean_jfi:.6f}±{std_jfi:.6f}\t{mean_final_ma_acc:.6f}±{std_final_ma_acc:.6f}"
        )

if __name__ == "__main__":
    main()
