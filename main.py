import os
import time
import argparse
import copy
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.multiprocessing import Manager, Process, Queue

from dataset import get_dataset, DatasetSplit
from models import get_model
from client import local_train
from server import aggregate
from test import test

from iot_device import IoTDevice
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict, Counter

import sys


def args_parser():
    parser = argparse.ArgumentParser()
    # federated learning arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--n_clients', type=int, default=40, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.9, help="learning rate decay")
    parser.add_argument('--lr_decay_step_size', type=int, default=500, help="step size to decay learning rate")

    # model and dataset arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or notc (default: non-iid)')
    parser.add_argument('--spc', action='store_true', help='whether spc or not (default: dirichlet)')
    parser.add_argument('--beta', type=float, default=0.5, help="beta for Dirichlet distribution")
    parser.add_argument('--n_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--n_channels', type=int, default=1, help="number of channels")

    # optimizing arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help="Optimizer (default: SGD)")
    parser.add_argument('--momentum', type=float, default=0.0, help="SGD momentum (default: 0.0)")
    parser.add_argument('--fed_strategy', type=str, default='fedavg', help="optimization scheme e.g. fedavg")
    parser.add_argument('--alpha', type=float, default=1.0, help="alpha for feddyn")

    # misc
    parser.add_argument('--n_gpu', type=int, default=4, help="number of GPUs")
    parser.add_argument('--n_procs', type=int, default=1, help="number of processes per processor")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--no_record', action='store_true', help='whether to record or not (default: record)')
    parser.add_argument('--load_checkpoint', action='store_true', help='whether to load model (default: do not load)')
    parser.add_argument('--use_checkpoint', action='store_true', help='whether to save best model (default: no checkpoint)')

    # UAV-FL
    parser.add_argument('--uavfl', action='store_true', help='for UAVFL simulation')
    parser.add_argument('--group_ratio', type=float, default=0.95, help="labels ratio for region group")
    parser.add_argument('--algorithm', type=str, default='random', help="algorithm selection (proposed, client_selection, random, pipeline)")    # 따로 추가
    parser.add_argument('--fedprox', action='store_true')
    parser.add_argument('--subchannels', type=int, default=3, help='number of subchannels')
    parser.add_argument('--n_clusters', type=int, default=3, help='number of clusters L for UAV waypoints')
    args = parser.parse_args()
    return args


def train_clients(args, param_queue, return_queue, device, train_dataset, client_settings):
    # seed setting
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # get model and train
    model = get_model(args=args, device=device)
    while True:
        # get message containing paramters
        param = param_queue.get()
        if param == "kill":
            # kill this process
            break
        else:
            # parameter setting
            lr = param['lr']
            sel_clients = param['sel_clients']
            c = param['c']

            # training multiple clients
            w_locals = []
            loss_locals = []
            c_locals = []
            for client in sel_clients:
                # get client settings
                setting = client_settings[client]
                c_i = setting.c_i
                K = setting.K
                # training dataloader for specific client
                dataloader = DataLoader(DatasetSplit(train_dataset, setting.dict_users), batch_size=args.local_bs, shuffle=True)
                # initialize model state dict
                model.load_state_dict(param['model_param'])
                # train a client
                w, loss, c_i, K = local_train(args, lr, c_i, c, K, model, dataloader, device)
                # append w, loss, lr, c_i, alpha
                w_locals.append(w)
                loss_locals.append(loss)
                if args.fed_strategy == 'scaffold':
                    c_locals.append(c_i)
                # modify settings
                setting.c_i = c_i
                setting.K = K
                del dataloader

            # return training results
            result = {'w_locals': w_locals, 'loss_locals': loss_locals, 'c_locals': c_locals}
            return_queue.put(result)
        del param
    del model


def zero_grad(model):
    grad = {k: torch.zeros(v.shape).cpu() for k, v in model.state_dict().items()}
    return grad


def dict_to_device(dict, device):
    for k in dict.keys():
        dict[k] = dict[k].detach().to(device)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # parse args and set seed
    args = args_parser()
    print("> Settings:", "epochs=",args.epochs, "n_clients=", args.n_clients, "algorithm=", args.algorithm, "dataset=", args.dataset, "model=", args.model, "iid=", args.iid, "localep=", args.local_ep)
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
    result_rootpath = './new_result'
    if not os.path.exists(result_rootpath):
        os.makedirs(result_rootpath)
    train_dataset, test_dataset, dict_users = get_dataset(args=args)
    global_model = get_model(args=args, device=devices[-1])
    if args.load_checkpoint:
        global_model.load_state_dict(torch.load(result_rootpath + '/{}_{}_L{}_C{}_{}_iid{}_spc{}.pt'.
                           format(args.dataset, args.model, args.local_ep, args.frac, args.fed_strategy, args.iid, args.spc)))
    w_glob = copy.deepcopy(global_model.state_dict())
    dict_to_device(w_glob, 'cpu')

    # create client setting list.
    manager = Manager()
    client_settings = []
    for idx in range(args.n_clients):
        s = manager.Namespace()
        s.dict_users = dict_users[idx]
        s.c_i = None
        s.K = 15
        client_settings.append(s)

    # set iot devices
    # Generate IoT device locations randomly in the region (0-400 x 0-400)
    region_data = np.random.uniform(0, 400, (args.n_clients, 2))

    iot_devices = [IoTDevice(x, y, len(dict_users[k]), np.random.uniform(150, 200, 1), args.dataset) for k, (x, y) in enumerate(region_data)]

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
        

    # create pool
    param_queues = []
    result_queues = []
    processes = []
    n_processes = n_devices * args.n_procs
    for i in range(n_devices):
        for _ in range(args.n_procs):
            param_queue, result_queue = Queue(), Queue()
            p = Process(target=train_clients, args=(args, param_queue, result_queue, devices[i], train_dataset, client_settings))
            p.start()
            processes.append(p)
            param_queues.append(param_queue)
            result_queues.append(result_queue)

    # start training
    client_all = list(range(args.n_clients))
    n_clients = int(args.frac * args.n_clients)
    c = zero_grad(global_model)
    lr = args.lr
    test_accs = []

    ########################################################################
    # Client selection
    ########################################################################

    MIN_BATTERY = 1.0
    MAX_COMM_TIME = 0.006

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
    
    
    for round in range(args.epochs):
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
        
        def select_clients_by_utility(feasible_clients, iot_devices):
            if len(feasible_clients) >= M:
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
            selected_clients = select_clients_by_utility(feasible_clients, iot_devices)
            print(f"[Round {round+1}] Utility: {len(selected_clients)} clients selected -> {selected_clients}")
            
        elif args.algorithm == 'pipelined':
            # Pipelined selection: select M clients for each cluster using utility-based method
            selected_clients = []
            cluster_ids = clusters.keys()
            
            for cid in cluster_ids:
                cluster_indices = clusters[cid]
                
                # Find feasible clients in this cluster
                feasible_in_cluster = list(set(cluster_indices).intersection(feasible_clients))
                if len(feasible_in_cluster) == 0:
                    print(f"Cluster {cid}: No feasible client. Skipping...")
                    continue
                
                # Select M clients from this cluster using utility-based selection
                chosen_indices = select_clients_by_utility(feasible_in_cluster, iot_devices)
                
                selected_clients.extend(chosen_indices)
            
            print(f"[Round {round+1}] Pipelined: {len(selected_clients)} clients selected -> {selected_clients}")
            
        else:  # Default to random selection
            if len(feasible_clients) >= M:
                selected_clients = np.random.choice(feasible_clients, size=M, replace=False).tolist()
            else:
                selected_clients = feasible_clients.copy()
            print(f"[Round {round+1}] Random: {len(selected_clients)} clients selected -> {selected_clients}")
        
        # Update last selected round for selected clients using IoT device methods
        for k in selected_clients:
            iot_devices[k].update_selection_round(round)


        ########################################################################
        # 3) federated learning
        ########################################################################
        clients = list(map(int, selected_clients))

        for idx, device in enumerate(iot_devices):
            if idx in clients:  # 선택된 클라이언트인지 확인
                comm_energy = device.get_comm_energy(device.get_commtime(uav_pos, model_param_size_bits, M))
                comp_energy = device.get_comp_energy()
                
                device.battery -= (comm_energy + comp_energy)  # 배터리 값 갱신


        
        # assign clients to processes
        assigned_clients = []
        n_assigned_client = J * M // n_processes
        #print(f"Assigned clients: {n_assigned_client} per process")
        for i in range(n_processes):
            assigned_clients.append(clients[:n_assigned_client])
            del clients[:n_assigned_client]
        for i, rest in enumerate(clients):
            assigned_clients[i].append(rest)
        # print(f"Assigned clients: {assigned_clients}")

        # start training
        start_time = time.time()
        for i in range(n_processes):
            param_queues[i].put({'model_param': copy.deepcopy(w_glob), 'lr': lr,
                                 'sel_clients': assigned_clients[i], 'c': copy.deepcopy(c)})

        # aggregate
        w_locals = []
        loss_locals = []
        c_locals = []
        for i in range(n_processes):
            result = result_queues[i].get()
            w_locals.extend(result['w_locals'])
            loss_locals.extend(result['loss_locals'])
            c_locals.extend(result['c_locals'])
        loss = sum(loss_locals) / len(loss_locals)
        
        # Update loss square for selected clients
        for i, client_idx in enumerate(selected_clients):
            if i < len(loss_locals):
                loss_square = loss_locals[i] ** 2
                iot_devices[client_idx].update_loss_square(loss_square)
        
        lr *= args.lr_decay ** (round // args.lr_decay_step_size)
        w_glob, c = aggregate(args, w_locals, w_glob, c, c_locals)
        print("Round {:3d} \t Training loss: {:.6f}".format(round + 1, loss), end=', ')
        del w_locals
        del loss_locals
        del c_locals

        # test
        global_model.load_state_dict(w_glob)
        test_acc, test_loss = test(args, global_model, test_dataset, devices[-1])
        test_accs.append(test_acc)
        print("Testing accuracy: {:.2f}, Time: {:.4f}".format(test_acc, time.time() - start_time))

        if args.use_checkpoint:
            if test_acc == max(test_accs):
                torch.save(w_glob, result_rootpath + '/{}_{}_L{}_C{}_{}_iid{}_spc{}.pt'.
                           format(args.dataset, args.model, args.local_ep, args.frac, args.fed_strategy, args.iid, args.spc))

    # close the pool to release resources
    for i in range(n_processes):
        param_queues[i].put("kill")

    time.sleep(5)
    for p in processes:
        p.join()
    
    # record test accuracies
    if not args.no_record:
        np.savetxt(f"./{result_rootpath}/{args.fedprox}_{args.beta}_{args.algorithm}_{args.n_clients}_{args.subchannels}_{args.dataset}.csv", test_accs, delimiter=",")
