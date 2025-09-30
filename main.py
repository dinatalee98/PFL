import os
import time
import copy
import random
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import get_dataset, DatasetSplit
from models import get_model
from client import local_train
from server import aggregate
from test import test

from iot_device import IoTDevice
from sklearn.cluster import KMeans
from tqdm import tqdm
from arguments import args_parser

def zero_grad(model):
    grad = {k: torch.zeros(v.shape).cpu() for k, v in model.state_dict().items()}
    return grad


def dict_to_device(dict, device):
    for k in dict.keys():
        dict[k] = dict[k].detach().to(device)


if __name__ == "__main__":
    args = args_parser()
    print(f"> Settings: n_clients={args.n_clients}, algorithm={args.algorithm}, dataset={args.dataset}, beta={args.beta}, subchannels={args.subchannels}, epochs={args.epochs}, localep={args.local_ep}")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # set device
    if torch.cuda.is_available():
        n_devices = min(torch.cuda.device_count(), args.n_gpu)
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
        cuda = True
        print("Using GPU")
    elif torch.backends.mps.is_available():
        n_devices = 1  # MPS는 단일 GPU만 지원
        devices = [torch.device("mps")]
        cuda = False
        print("Using MPS (Apple GPU)")
    else:
        n_devices = 1
        devices = [torch.device('cpu')]
        cuda = False
    os.environ["OMP_NUM_THREADS"] = "1"
    num_processes = torch.multiprocessing.cpu_count()  # Number of available CPU cores

    # create dataset and model
    result_rootpath = args.result_path
    if not os.path.exists(result_rootpath):
        os.makedirs(result_rootpath)
    train_dataset, test_dataset, dict_users = get_dataset(args=args)
    global_model = get_model(args=args, device=devices[-1])
    if args.load_checkpoint:
        global_model.load_state_dict(torch.load(result_rootpath + '/{}_{}_L{}_C{}_{}_iid{}_spc{}.pt'.
                           format(args.dataset, args.model, args.local_ep, args.frac, args.fed_strategy, args.iid, args.spc)))
    w_glob = copy.deepcopy(global_model.state_dict())
    dict_to_device(w_glob, 'cpu')
    
    
    result_file = open(f"./{result_rootpath}/{args.dataset}_{args.algorithm}_{args.n_clients}_{args.beta}_{args.subchannels}_{args.lambda_stale}.txt", "a")
    result_file.write(f"round, test_acc, test_loss, selected_clients\n")


    # set iot devices
    # Generate IoT device locations randomly in the region (0-400 x 0-400)
    region_data = np.random.uniform(0, 400, (args.n_clients, 2))

    iot_devices = [IoTDevice(x, y, len(dict_users[k]), np.random.uniform(150, 200, 1), args.dataset, args.lambda_stale) for k, (x, y) in enumerate(region_data)]

    comp_times = np.array([device.get_computation_time() for device in iot_devices]).flatten()
    
    print(f"Computation times: {comp_times}")

    M = args.subchannels # Number of subchannels
    J = 1  # Number of clusters


    if args.algorithm == 'proposed':
        tau = np.std(comp_times) * 2

        # Sort devices in ascending order of compute times
        sorted_indices = np.argsort(comp_times)
        sorted_times = comp_times[sorted_indices]
        
        # Determine number of groups J using equation (2)
        t_min, t_max = np.min(sorted_times), np.max(sorted_times)
        if tau <= 0:
            J = 1
        else:
            J_float = (t_max - t_min) / tau
            J = int(np.ceil(J_float)) if J_float > 1 else 1
        
        print(f"[Latency-Aware Grouping] Computed number of groups J = {J} (tau={tau}, K={args.n_clients})")
        
        # Compute base group size and remainder
        K = args.n_clients
        base_size = K // J
        remainder = K % J
        
        group_sizes = [base_size + 1 if j < remainder else base_size for j in range(J) ]
        
        # Initialize index and construct groups
        clusters = {}
        i = 0
        for j, n_j in enumerate(group_sizes):
            clusters[j] = sorted_indices[i:i+n_j].tolist()
            i += n_j
        
        # Print final clusters
        for cid, devs in clusters.items():
            print(f"Group {cid}: {len(devs)} devices -> {devs}")
            pass
        

    # get model for training
    model = get_model(args=args, device=devices[-1])

    # start training
    client_all = list(range(args.n_clients))
    c = zero_grad(global_model)
    lr = args.lr
    test_accs = []

    ########################################################################
    # Client selection
    ########################################################################

    if args.dataset == 'mnist':
        MIN_BATTERY = 1.0
        MAX_COMM_TIME = 0.007
    elif args.dataset == 'cifar10':
        MIN_BATTERY = 1.0
        MAX_COMM_TIME = 0.2


    # Cluster IoT devices by geographical location using K-means
    device_locations = np.array([[device.x, device.y] for device in iot_devices])
    L = args.n_clusters  # Number of clusters
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=L, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(device_locations)
    
    # Calculate cluster centroids as waypoints
    cluster_centroids = kmeans.cluster_centers_
    waypoints = np.column_stack([cluster_centroids, np.full(L, 100.0)])  # Add z=100.0 for all waypoints
    
    print(f"Clustered {len(iot_devices)} IoT devices into {L} clusters")
    print(f"Waypoints (centroids): {waypoints}")
    
    # Greedy algorithm for waypoint ordering (nearest neighbor)
    def greedy_waypoint_ordering(waypoints):
        if len(waypoints) <= 1:
            return waypoints
        
        ordered_waypoints = []
        remaining_waypoints = waypoints.copy()
        
        # Start with an arbitrary waypoint (first one)
        current_waypoint = remaining_waypoints[0]
        ordered_waypoints.append(current_waypoint)
        remaining_waypoints = np.delete(remaining_waypoints, 0, axis=0)
        
        # Greedy selection: always choose the nearest remaining waypoint
        while len(remaining_waypoints) > 0:
            distances = [np.linalg.norm(current_waypoint - wp) for wp in remaining_waypoints]
            nearest_idx = np.argmin(distances)
            current_waypoint = remaining_waypoints[nearest_idx]
            ordered_waypoints.append(current_waypoint)
            remaining_waypoints = np.delete(remaining_waypoints, nearest_idx, axis=0)
        
        return np.array(ordered_waypoints)
    
    # Order waypoints using greedy algorithm
    ordered_waypoints = greedy_waypoint_ordering(waypoints)
    print(f"Ordered waypoints: {ordered_waypoints}")
    
    
    epsilon = args.epsilon_start
    for round in tqdm(range(args.epochs), dynamic_ncols=True):
        # Move UAV to waypoints in order using greedy algorithm
        waypoint_index = round % len(ordered_waypoints)
        uav_pos = ordered_waypoints[waypoint_index].copy()

        ########################################################################
        # 1) feasible clients check
        ########################################################################
        feasible_clients = []

        # Example: compute model parameter size (in bits)
        model_param_count = sum(p.numel() for p in global_model.parameters())
        model_param_size_bits = model_param_count * 32  # float32 => 32 bits
        
        for k in range(args.n_clients):
            battery_ok = (iot_devices[k].get_battery() >= MIN_BATTERY)
            comm_ok = (iot_devices[k].get_commtime(uav_pos, model_param_size_bits, M) <= MAX_COMM_TIME)
            
            if battery_ok and comm_ok:
                feasible_clients.append(k)

        if len(feasible_clients) == 0:
            print(f"Round {round+1}: No feasible client. Skipping...")
            continue
        

        ########################################################################
        # 2) Client selection based on algorithm
        ########################################################################
        
        def select_clients_by_utility(feasible_clients):
            if len(feasible_clients) >= M:
                if np.random.rand() > epsilon:
                    selected_clients = np.random.choice(feasible_clients, size=M, replace=False).tolist()
                else:
                    recency_utilities = []
                    for k in feasible_clients:
                        recency_util = iot_devices[k].compute_utility_with_stale_term(round)
                        recency_utilities.append(recency_util)
                    
                    recency_utilities = np.array(recency_utilities)
                    sorted_indices = np.argsort(recency_utilities)[::-1]  # Descending order
                    top_M_indices = sorted_indices[:M]
                    selected_clients = [feasible_clients[idx] for idx in top_M_indices]
            else:
                selected_clients = feasible_clients.copy()
            
            return selected_clients

        
        if args.algorithm == 'utility':
            selected_clients = select_clients_by_utility(feasible_clients)
            # print(f"[Round {round+1}] Utility: {len(selected_clients)} clients selected -> {selected_clients}")
            
        elif args.algorithm == 'proposed':
            # Pipelined selection: select M clients for each cluster using utility-based method
            selected_clients = []
            cluster_ids = clusters.keys()
            
            for cid in cluster_ids:
                cluster_indices = clusters[cid]
                
                # Find feasible clients in this cluster
                feasible_in_cluster = list(set(cluster_indices).intersection(feasible_clients))
                if len(feasible_in_cluster) == 0:
                    # print(f"Cluster {cid}: No feasible client. Skipping...")
                    continue
                
                # Select M clients from this cluster using utility-based selection
                chosen_indices = select_clients_by_utility(feasible_in_cluster)
                
                selected_clients.extend(chosen_indices)
            
            # print(f"[Round {round+1}] Proposed: {len(selected_clients)} clients selected -> {selected_clients}")
            
        else:  # Default to random selection
            if len(feasible_clients) >= M:
                selected_clients = np.random.choice(feasible_clients, size=M, replace=False).tolist()
            else:
                selected_clients = feasible_clients.copy()
            # print(f"[Round {round+1}] Random: {len(selected_clients)} clients selected -> {selected_clients}")
        
        # Update last selected round for selected clients using IoT device methods
        for k in selected_clients:
            iot_devices[k].update_selection_round(round)
        epsilon *= args.epsilon_decay
        epsilon = max(args.epsilon_min, epsilon)


        ########################################################################
        # 3) federated learning
        ########################################################################
        clients = list(map(int, selected_clients))

        for idx, client_idx in enumerate(selected_clients):
            comm_energy = iot_devices[client_idx].get_comm_energy(iot_devices[client_idx].get_commtime(uav_pos, model_param_size_bits, M))
            comp_energy = iot_devices[client_idx].get_comp_energy()
            
            iot_devices[client_idx].battery -= (comm_energy + comp_energy)


        
        # start training
        start_time = time.time()
        w_locals = []
        loss_locals = []
        c_locals = []
        
        for client in clients:
            # training dataloader for specific client
            dataloader = DataLoader(DatasetSplit(train_dataset, dict_users[client]), batch_size=args.local_bs, shuffle=True)
            # initialize model state dict
            model.load_state_dict(w_glob)
            # train a client
            w, loss, c_i, K = local_train(args, lr, None, c, 0, model, dataloader, devices[-1])
            # append w, loss, lr, c_i, alpha
            w_locals.append(w)
            loss_locals.append(loss)
            if args.fed_strategy == 'scaffold':
                c_locals.append(c_i)
            del dataloader
        
        loss = sum(loss_locals) / len(loss_locals)
        
        # Update loss square for selected clients
        for i, client_idx in enumerate(selected_clients):
            loss_square = loss_locals[i] ** 2

            # U_k = iot_devices[client_idx].num_of_data * math.sqrt(loss_square / iot_devices[client_idx].num_of_data)
            # print(f"Round {round+1} Client {client_idx} utility: {U_k}, loss square: {loss_square}")

            iot_devices[client_idx].update_loss_square(loss_square)
        
        # lr *= args.lr_decay ** (round // args.lr_decay_step_size)
        w_glob, c = aggregate(args, w_locals, w_glob, c, c_locals)
        # print("Round {:3d} \t Training loss: {:.6f}".format(round + 1, loss), end=', ')
        del w_locals
        del loss_locals
        del c_locals

        
        # test
        global_model.load_state_dict(w_glob)
        test_acc, test_loss = test(args, global_model, test_dataset, devices[-1])
        test_accs.append(test_acc)
        # print("Testing accuracy: {:.2f}".format(test_acc))
        
        result_file.write(f"{round+1}, {test_acc:.10f}, {test_loss:.10f}, {selected_clients}\n")
        result_file.flush()
    