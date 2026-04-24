import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--n_clients', type=int, default=40, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.9, help="learning rate decay")
    parser.add_argument('--lr_decay_step_size', type=int, default=500, help="step size to decay learning rate")
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or notc (default: non-iid)')
    parser.add_argument('--spc', action='store_true', help='whether spc or not (default: dirichlet)')
    parser.add_argument('--beta', type=float, default=0.5, help="beta for Dirichlet distribution")
    parser.add_argument('--n_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--n_channels', type=int, default=1, help="number of channels")
    parser.add_argument('--optimizer', type=str, default='sgd', help="Optimizer (default: SGD)")
    parser.add_argument('--momentum', type=float, default=0.0, help="SGD momentum (default: 0.0)")
    parser.add_argument('--fed_strategy', type=str, default='fedavg', help="optimization scheme e.g. fedavg")
    parser.add_argument('--alpha', type=float, default=1.0, help="alpha for feddyn")
    parser.add_argument('--n_gpu', type=int, default=4, help="number of GPUs")
    parser.add_argument('--n_procs', type=int, default=1, help="number of processes per processor")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--no_record', action='store_true', help='whether to record or not (default: record)')
    parser.add_argument('--load_checkpoint', action='store_true', help='whether to load model (default: do not load)')
    parser.add_argument('--use_checkpoint', action='store_true', help='whether to save best model (default: no checkpoint)')
    parser.add_argument('--uavfl', action='store_true', help='for UAVFL simulation')
    parser.add_argument('--group_ratio', type=float, default=0.95, help="labels ratio for region group")
    parser.add_argument('--algorithm', type=str, default='random', help="algorithm selection (proposed, client_selection, random, pipeline)")
    parser.add_argument('--fedprox', action='store_true')
    parser.add_argument('--subchannels', type=int, default=3, help='number of subchannels')
    parser.add_argument('--n_clusters', type=int, default=3, help='number of clusters L for UAV waypoints')
    parser.add_argument("--window", type=int, default=10, help='window size for moving average')
    parser.add_argument("--result_path", type=str, default='result', help='result path')
    parser.add_argument("--source_folder", type=str, help='source folder containing training result files')
    parser.add_argument('--epsilon_start', type=float, default=0.9, help='epsilon-greedy start value')
    parser.add_argument('--epsilon_decay', type=float, default=0.98, help='epsilon decay per round')
    parser.add_argument('--epsilon_min', type=float, default=0.2, help='minimum epsilon')
    parser.add_argument('--lambda_stale', type=float, default=0.2, help='lambda stale term weight')
    parser.add_argument(
        '--lambda_stale_values',
        type=float,
        nargs='+',
        default=None,
        help='specific lambda_stale values to plot (e.g. --lambda_stale_values 0.0 0.1 0.2)',
    )
    parser.add_argument('--max_round', type=int, default=None, help='maximum round to plot (default: plot all rounds)')
    parser.add_argument('--target_accuracy', type=float, default=None, help='target accuracy to inspect (default: None)')
    parser.add_argument('--tau', type=float, default=0.1)
    args = parser.parse_args()
    return args


