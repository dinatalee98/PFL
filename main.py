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


def args_parser():
    parser = argparse.ArgumentParser()
    # federated learning arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--n_clients', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
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
    parser.add_argument('--beta', type=float, default=0.2, help="beta for Dirichlet distribution")
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
    parser.add_argument('--uavfl', action='store_true', default=True, help='for UAVFL simulation')
    parser.add_argument('--group_ratio', type=float, default=0.95, help="labels ratio for region group")
    parser.add_argument('--algorithm', type=str, default='proposed', help="algorithm selection (proposed, speed, random)")

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
    print("> Settings:", "epochs=",args.epochs, "n_clients=", args.n_clients, "algorithm=", args.algorithm, "dataset=", args.dataset, "model=", args.model, "iid=", args.iid)
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
    else:
        n_devices = 1
        devices = [torch.device('cpu')]
        cuda = False
    os.environ["OMP_NUM_THREADS"] = "1"
    num_processes = torch.multiprocessing.cpu_count()  # Number of available CPU cores

    # create dataset and model
    result_rootpath = './result'
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
    unit_num = int(args.n_clients/4)
    region1 = np.random.uniform(0, 10, (2*unit_num, 2))
    region2 = np.random.uniform(20, 30, (unit_num, 2))
    region3 = np.random.uniform(40, 50, (args.n_clients-3*unit_num, 2))
    region_data = np.vstack((region1, region2, region3))

    num_of_data = np.array([len(dict_users[k]) for k in range(args.n_clients)])
    iot_devices = [IoTDevice(x, y, num_of_data, np.random.uniform(1, 30, 1)) for (x, y) in region_data]
    
    comp_times = np.array([device.get_computation_time() for device in iot_devices]).flatten()


    M = 10 # Number of subchannels
    J = 1  # Number of clusters

    if args.algorithm == 'proposed':
        ########################################################################
        # Example: Pipeline-based clustering by local computation time
        #
        # 1) Sort devices by comp_times
        # 2) Determine number of clusters J from the range of comp_times and tau
        # 3) Group them in ascending order, ensuring each cluster has at least M devices
        # 4) Store the result in a dictionary called `clusters`
        ########################################################################

        # Decide your 'tau' (time slot)
        # For demonstration, we'll define them as constants below.
        # Feel free to move them to args_parser() or set them in other ways.
        tau = 5.0     # Example: time-slot length (seconds) or your own chosen unit

        # Sort clients by their compute times
        sorted_indices = np.argsort(comp_times)       # Indices of devices sorted by ascending compute time
        sorted_times = comp_times[sorted_indices]
        
        # Determine number of clusters, J
        t_min, t_max = np.min(sorted_times), np.max(sorted_times)
        # Safeguard against division by zero if tau <= 0
        if tau <= 0:
            J = 1
        else:
            # Example rule: number of clusters ~ (t_max - t_min) / tau
            J_float = (t_max - t_min) / tau
            J = int(np.ceil(J_float)) if J_float > 1 else 1
        
        #print(f"[Pipeline Clustering] Computed number of clusters J = {J} (tau={tau}, M={M})")
        
        # Balanced cluster size if you want to keep them roughly uniform
        # but with an additional constraint that each cluster has at least M devices
        n_ideal = max(1, args.n_clients // J)
        
        ########################################################################
        # Construct the clusters
        ########################################################################
        clusters = {}   # dictionary: cluster_id -> list of device indices
        current_cluster_id = 0
        
        i = 0
        while i < args.n_clients and current_cluster_id < J:
            # Start a new cluster
            clusters[current_cluster_id] = []
            cluster_size = 0
            
            # Fill up to n_ideal
            while cluster_size < n_ideal and i < args.n_clients:
                clusters[current_cluster_id].append(sorted_indices[i])
                i += 1
                cluster_size += 1
            
            # If this new cluster hasn't reached M devices, try adding more
            while cluster_size < M and i < args.n_clients:
                clusters[current_cluster_id].append(sorted_indices[i])
                i += 1
                cluster_size += 1
            
            current_cluster_id += 1
        
        # If leftover devices remain, assign them to whichever cluster(s) you like
        # for balanced distribution or simply to the last cluster:
        while i < args.n_clients:
            clusters[current_cluster_id - 1].append(sorted_indices[i])
            i += 1
        
        # Print final clusters
        for cid, devs in clusters.items():
            #print(f"Cluster {cid}: {len(devs)} devices -> {devs}")
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

    MIN_BATTERY = 10.0
    MAX_COMM_TIME = 5.0

    uav_pos = np.array([0.0, 0.0, 100.0])  # (x, y, z)

    for round in range(args.epochs):
        ########################################################################
        # 1) 각 클라이언트별 "현재 글로벌 모델 기준" 유틸리티 계산
        ########################################################################
        #  - iot_devices[k].get_local_dataset() 등으로 해당 클라이언트 데이터 획득
        #  - compute_pretraining_utility(...) 호출
        #  - 결과를 배열 or dict에 저장
        ########################################################################
        utilities = [0.0] * args.n_clients
        global_model.load_state_dict(w_glob)   # 직전 라운드까지 집계된 글로벌 모델
        for k in range(args.n_clients):
            local_indices = list(dict_users[k])
            local_dataset_k = DataLoader(DatasetSplit(train_dataset, local_indices), batch_size=args.local_bs, shuffle=True)
            utilities[k] = iot_devices[k].compute_pretraining_utility(
                global_model, local_dataset_k, device=devices[-1]
            )
        ########################################################################
        # 1) 모든 클라이언트에 대한 feasibility check (공통)
        ########################################################################
        feasible_clients = []

        # UAV 움직임
        # 예: uav_pos = np.array([x, y, z])
        # Example: compute model parameter size (in bits)
        model_param_count = sum(p.numel() for p in global_model.parameters())
        model_param_size_bits = model_param_count * 32  # float32 => 32 bits

        for k in range(args.n_clients):
            battery_ok = (iot_devices[k].get_battery() >= MIN_BATTERY)   # 예: 배터리 임계값
            comm_ok    = (iot_devices[k].get_commtime(uav_pos, model_param_size_bits)  <= MAX_COMM_TIME)  # 예: 통신시간 임계값
            if battery_ok and comm_ok:
                feasible_clients.append(k)

        # 만약 하나도 feasible 하지 않다면, 이번 라운드는 그냥 건너뛴다거나 하는 처리
        if len(feasible_clients) == 0:
            print(f"Round {round+1}: No feasible client. Skipping...")
            # 원하는 대로 처리 (continue 등)
            continue

        ########################################################################
        # 2) Selection 방식에 따른 분기 (Proposed / Speed / Random 등)
        ########################################################################
        if args.algorithm == 'proposed':
            # (a) 미리 만들어놓은 파이프라인 클러스터(예: compute-time 기반)
            #     clusters: dict -> cluster_id -> list_of_device_indices
            # (b) 각 클러스터별로 feasible_clients와의 교집합만 뽑기
            # (c) Utility + Epsilon-Greedy 등 적용

            # 예시: n_clients = int(args.frac * args.n_clients)
            n_clients = M  # 예시: M = 10
            cluster_ids = clusters.keys()
            """
            print(f"[Round {round+1}] Proposed: {len(cluster_ids)} clusters -> {cluster_ids}")
            leftover = n_clients % len(cluster_ids)
            base_size = n_clients // len(cluster_ids)
            """

            # epsilon 갱신용
            if 'epsilon' not in locals():
                epsilon = 1.0
            epsilon_min   = 0.1
            epsilon_decay = 0.95

            selected_clients = []

            for cid in cluster_ids:
                # 2-1) 클러스터 내 클라이언트
                cluster_indices = clusters[cid]

                # 2-2) 'feasible_clients'와 교집합
                feasible_in_cluster = list(set(cluster_indices).intersection(feasible_clients))
                if len(feasible_in_cluster) == 0:
                    continue

                # 2-4) 유틸리티 계산 (예: 랜덤 예시)
                # (c) "학습 전"에 계산한 유틸리티 배열 생성
                U_array = np.array([ utilities[idx] for idx in feasible_in_cluster ])
                U_sum   = np.sum(U_array) if np.sum(U_array) > 0 else 1e-12
                # 2-5) Epsilon-Greedy 선택
                chosen_indices = []
                idx_order = np.random.permutation(len(feasible_in_cluster))
                if epsilon < np.random.rand():
                    # Randomly select M elements from feasible_in_cluster
                    if len(feasible_in_cluster) >= M:
                        chosen_indices = np.random.choice(feasible_in_cluster, size=M, replace=False)
                    else:
                        chosen_indices = feasible_in_cluster
                else:
                    # U_k가 큰 상위 M개 요소 선택
                    if len(feasible_in_cluster) >= M:
                        # U_array와 feasible_in_cluster를 함께 정렬
                        sorted_indices = np.argsort(U_array)[::-1]  # 내림차순 정렬
                        top_M_indices = sorted_indices[:M]
                        chosen_indices = [feasible_in_cluster[idx] for idx in top_M_indices]
                    else:
                        chosen_indices = feasible_in_cluster

                selected_clients.extend(chosen_indices)

            # epsilon decay
            epsilon = max(epsilon_min, epsilon*epsilon_decay)

            # print(f"[Round {round+1}] Proposed: {len(selected_clients)} clients selected -> {selected_clients}")

            # >>> 이후 local_train, aggregation 등에 selected_clients 사용

        elif args.algorithm == 'speed':
            # Speed Selection 예시:
            #  - 이미 KMeans 등으로 speed-based cluster를 구했다고 가정
            #  - 혹은 단순히 comp_times가 작은 순으로 pick
            #  - 이때도 feasible_clients만 대상으로 함

            # 예: comp_times가 작은 순서대로 feasible_clients를 정렬한 뒤 n_clients만큼 뽑기
            feasible_comp_times = [(k, comp_times[k]) for k in feasible_clients]
            feasible_comp_times.sort(key=lambda x: x[1])  # ascending
            selected_clients = [x[0] for x in feasible_comp_times[:M]]

            print(f"[Round {round+1}] Speed: {len(selected_clients)} clients selected -> {selected_clients}")

            # >>> 이후 local_train, aggregation 등에 selected_clients 사용

        else:
            # Random Selection (IID):
            #  - feasible_clients 중에서 n_clients 뽑기
            selected_clients = np.random.choice(feasible_clients, size = M, replace=False)
            print(f"[Round {round+1}] Random: {len(selected_clients)} -> {selected_clients}")

        ########################################################################
        # 3) 실제 로컬 훈련, 파라미터 큐 전송, 결과 수신, 모델 집계 등
        ########################################################################
        # 예: param_queue에 model_param, lr, sel_clients=selected_clients 등을 보내고,
        #     result_queues에서 w_locals, loss_locals 수신한 뒤 aggregate
        clients = list(map(int, selected_clients))

        # assign clients to processes
        assigned_clients = []
        n_assigned_client = J * n_clients // n_processes
        print(f"Assigned clients: {n_assigned_client} per process")
        for i in range(n_processes):
            assigned_clients.append(clients[:n_assigned_client])
            del clients[:n_assigned_client]
        for i, rest in enumerate(clients):
            assigned_clients[i].append(rest)

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
        np.savetxt(f"./result/{args.algorithm}_{args.n_clients}_{args.frac}.csv", test_accs, delimiter=",")
