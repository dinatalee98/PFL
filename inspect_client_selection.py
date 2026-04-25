import os
import random
import torch
import numpy as np

from dataset import get_dataset
from models import get_model
from iot_device import IoTDevice
from sklearn.cluster import KMeans
from arguments import args_parser


def greedy_waypoint_ordering(waypoints):
    if len(waypoints) <= 1:
        return waypoints
    ordered_waypoints = []
    remaining_waypoints = waypoints.copy()
    current_waypoint = remaining_waypoints[0]
    ordered_waypoints.append(current_waypoint)
    remaining_waypoints = np.delete(remaining_waypoints, 0, axis=0)
    while len(remaining_waypoints) > 0:
        distances = [np.linalg.norm(current_waypoint - wp) for wp in remaining_waypoints]
        nearest_idx = np.argmin(distances)
        current_waypoint = remaining_waypoints[nearest_idx]
        ordered_waypoints.append(current_waypoint)
        remaining_waypoints = np.delete(remaining_waypoints, nearest_idx, axis=0)
    return np.array(ordered_waypoints)


if __name__ == "__main__":
    args = args_parser()
    print(f"> Settings: n_clients={args.n_clients}, algorithm={args.algorithm}, dataset={args.dataset}, beta={args.beta}, subchannels={args.subchannels}, epochs={args.epochs}, localep={args.local_ep}, lambda_stale={args.lambda_stale}, tau={args.tau}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    if torch.cuda.is_available():
        n_devices = min(torch.cuda.device_count(), args.n_gpu)
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
    elif torch.backends.mps.is_available():
        devices = [torch.device("mps")]
    else:
        devices = [torch.device("cpu")]

    result_rootpath = args.result_path
    if not os.path.exists(result_rootpath):
        os.makedirs(result_rootpath)

    _, _, dict_users = get_dataset(args=args)
    global_model = get_model(args=args, device=devices[-1])
    if args.load_checkpoint:
        global_model.load_state_dict(torch.load(result_rootpath + '/{}_{}_L{}_C{}_{}_iid{}_spc{}.pt'.
                           format(args.dataset, args.model, args.local_ep, args.frac, args.fed_strategy, args.iid, args.spc)))

    region_data = np.random.uniform(0, 400, (args.n_clients, 2))
    c_k = 3 * 3 * 10**4 if args.dataset == "cifar10" else 3 * 10**4
    model_param_count = sum(p.numel() for p in global_model.parameters())
    model_param_size_bits = model_param_count * 32
    iot_devices = [IoTDevice(x, y, len(dict_users[k]), c_k, model_param_size_bits, args.subchannels, args.lambda_stale, args.local_ep) for k, (x, y) in enumerate(region_data)]

    comp_times = np.array([device.get_comp_time() for device in iot_devices]).flatten()
    M = args.subchannels

    clusters = {}
    if args.algorithm == "proposed" or args.algorithm == "pipeline":
        sorted_indices = np.argsort(comp_times)
        sorted_times = comp_times[sorted_indices]
        t_min, t_max = np.min(sorted_times), np.max(sorted_times)
        if args.tau <= 0:
            J = 1
        else:
            J_float = (t_max - t_min) / args.tau
            J = int(np.ceil(J_float)) if J_float > 1 else 1
        K = args.n_clients
        base_size = K // J
        remainder = K % J
        group_sizes = [base_size + 1 if j < remainder else base_size for j in range(J)]
        i = 0
        for j, n_j in enumerate(group_sizes):
            clusters[j] = sorted_indices[i:i + n_j].tolist()
            i += n_j

    MAX_COMM_TIME = args.tau
    device_locations = np.array([[device.x, device.y] for device in iot_devices])
    L = args.n_clusters
    kmeans = KMeans(n_clusters=L, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(device_locations)
    cluster_centroids = kmeans.cluster_centers_
    waypoints = np.column_stack([cluster_centroids, np.full(L, 100.0)])
    ordered_waypoints = greedy_waypoint_ordering(waypoints)

    result_file = open(f"./{result_rootpath}/{args.dataset}_{args.algorithm}_{args.n_clients}_{args.beta}_{args.subchannels}_{args.lr}_{args.lambda_stale}_{args.tau}_{args.seed}_selection_inspection.txt", "a")
    result_file.write("round, selected_clients_count, selected_clients\n")
    epsilon = args.epsilon_start

    def select_clients_by_utility(available_clients, current_round):
        if len(available_clients) >= M:
            if np.random.rand() > epsilon:
                return np.random.choice(available_clients, size=M, replace=False).tolist()
            recency_utilities = []
            for k in available_clients:
                recency_utilities.append(iot_devices[k].compute_utility(current_round))
            recency_utilities = np.array(recency_utilities)
            sorted_indices = np.argsort(recency_utilities)[::-1]
            top_m_indices = sorted_indices[:M]
            return [available_clients[idx] for idx in top_m_indices]
        return available_clients.copy()

    for round in range(args.epochs):
        waypoint_index = round % len(ordered_waypoints)
        uav_pos = ordered_waypoints[waypoint_index].copy()
        communication_available_clients = []
        for k in range(args.n_clients):
            comm_ok = iot_devices[k].get_comm_time(uav_pos) <= MAX_COMM_TIME
            if comm_ok:
                communication_available_clients.append(k)

        if args.algorithm == "utility":
            selected_clients = select_clients_by_utility(communication_available_clients, round)
        elif args.algorithm == "proposed" or args.algorithm == "pipeline":
            selected_clients = []
            for cid in clusters.keys():
                cluster_indices = clusters[cid]
                available_in_cluster = list(set(cluster_indices).intersection(communication_available_clients))
                if len(available_in_cluster) == 0:
                    continue
                if args.algorithm == "proposed":
                    chosen_indices = select_clients_by_utility(available_in_cluster, round)
                else:
                    if len(available_in_cluster) >= M:
                        chosen_indices = np.random.choice(available_in_cluster, size=M, replace=False).tolist()
                    else:
                        chosen_indices = available_in_cluster.copy()
                selected_clients.extend(chosen_indices)
        else:
            if len(communication_available_clients) >= M:
                selected_clients = np.random.choice(communication_available_clients, size=M, replace=False).tolist()
            else:
                selected_clients = communication_available_clients.copy()

        for k in selected_clients:
            iot_devices[k].last_selected_round = round + 1
        epsilon *= args.epsilon_decay
        epsilon = max(args.epsilon_min, epsilon)

        result_file.write(f"{round + 1}, {len(selected_clients)}, {selected_clients}\n")
        result_file.flush()

    result_file.close()
