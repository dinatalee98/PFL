import ast
import os
import re
from collections import Counter
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    seed: int


FILENAME_RE = re.compile(
    r"^(?P<dataset>[^_]+)_(?P<algorithm>[^_]+)_(?P<n_clients>\d+)"
    r"_(?P<beta>-?\d+(?:\.\d+)?)_(?P<subchannels>\d+)"
    r"_(?P<lr>-?\d+(?:\.\d+)?)_(?P<lambda_stale>-?\d+(?:\.\d+)?)_(?P<seed>\d+)\.txt$"
)


def parse_run_key(filename: str) -> RunKey | None:
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    g = m.groupdict()
    return RunKey(
        dataset=g["dataset"],
        algorithm=g["algorithm"],
        n_clients=int(g["n_clients"]),
        beta=float(g["beta"]),
        subchannels=int(g["subchannels"]),
        lr=float(g["lr"]),
        lambda_stale=float(g["lambda_stale"]),
        seed=int(g["seed"]),
    )


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


def mean_curves(all_rounds: list[list[int]], all_accs: list[list[float]]) -> tuple[list[int], list[float]]:
    if not all_rounds:
        return [], []
    if len(all_rounds) == 1:
        return all_rounds[0], all_accs[0]
    common = sorted(set(all_rounds[0]).intersection(*[set(r) for r in all_rounds[1:]]))
    if not common:
        common = sorted(set(all_rounds[0]))
        all_interp = []
        for rs, ac in zip(all_rounds, all_accs):
            interp = np.interp(common, rs, ac).tolist()
            all_interp.append(interp)
    else:
        all_interp = []
        for rs, ac in zip(all_rounds, all_accs):
            idx_map = {r: a for r, a in zip(rs, ac)}
            all_interp.append([idx_map[r] for r in common])
    avg = np.mean(all_interp, axis=0).tolist()
    return common, avg


def main() -> None:
    from arguments import args_parser
    args = args_parser()

    out = getattr(args, "out", None)
    root = getattr(args, "source_folder", None) or getattr(args, "result_path", "result")
    selected_lambdas = getattr(args, "lambda_stale_values", [])
    if not os.path.isdir(root):
        raise SystemExit(f"result_path not found: {root}")

    series: dict[float, list[tuple[list[int], list[float], RunKey, Counter]]] = {}
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
        if not is_selected_lambda(key.lambda_stale, selected_lambdas):
            continue

        path = os.path.join(root, name)
        rounds, accs, participation = read_round_acc(path, args.max_round)
        if not rounds:
            continue
        series.setdefault(key.lambda_stale, []).append((rounds, accs, key, participation))

    if not series:
        raise SystemExit("no matching result files found")

    keys_sorted = sorted(series.keys())
    print("lambda_stale\tseeds\tAUC (mean±std)\t\tJain_Fairness (mean±std)")
    for lam in keys_sorted:
        entries = series[lam]
        aucs = []
        jfis = []
        for rounds, accs, key, participation in entries:
            if len(rounds) >= 2:
                auc = float(np.trapezoid(accs, rounds) if hasattr(np, "trapezoid") else np.trapz(accs, rounds))
            else:
                auc = float("nan")
            jfi = jain_fairness(participation, key.n_clients)
            aucs.append(auc)
            jfis.append(jfi)
        mean_auc = float(np.nanmean(aucs))
        std_auc = float(np.nanstd(aucs))
        mean_jfi = float(np.nanmean(jfis))
        std_jfi = float(np.nanstd(jfis))
        print(f"{lam:g}\t{len(entries)}\t{mean_auc:.4f}±{std_auc:.4f}\t\t{mean_jfi:.6f}±{std_jfi:.6f}")

    first_key = series[keys_sorted[0]][0][2]
    title_bits = [f"{first_key.dataset}", f"{first_key.algorithm}", f"K={first_key.n_clients}", f"beta={first_key.beta}", f"M={first_key.subchannels}", f"lr={first_key.lr}"]
    title_bits.append(f"seeds={len(series[keys_sorted[0]])}")

    plt.figure(figsize=(10, 6))
    for lam in keys_sorted:
        entries = series[lam]
        all_rounds = [e[0] for e in entries]
        all_accs = [e[1] for e in entries]
        avg_rounds, avg_accs = mean_curves(all_rounds, all_accs)
        w = int(getattr(args, "window", 1) or 1)
        smoothed = moving_average(avg_accs, w)
        x = avg_rounds if len(smoothed) == len(avg_accs) else avg_rounds[w - 1 :]
        plt.plot(x, smoothed, linewidth=2, label=f"lambda_stale={lam:g}")

    plt.xlabel("round")
    plt.ylabel("test accuracy")
    plt.title(" | ".join(title_bits))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if out is None:
        out = os.path.join(root, "lambda_stale_plot.png")
    plt.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
