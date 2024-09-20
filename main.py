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

np.random.seed(1)



def args_parser():
    parser = argparse.ArgumentParser()
    # federated learning arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--n_clients', type=int, default=120, help="number of users: K")
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
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
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
    print("> Settings: ", args)
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
    region1 = np.random.uniform(0, 10, (int(args.n_clients/2), 2))
    region2 = np.random.uniform(20, 30, (int(args.n_clients/4), 2))
    region3 = np.random.uniform(40, 50, (int(args.n_clients/4), 2))
    region_data = np.vstack((region1, region2, region3))

    iot_devices = [IoTDevice(x, y, np.random.uniform(1, 30, 1)) for (x, y) in region_data]
    comp_times = np.array([device.get_computation_time() for device in iot_devices])

    if args.algorithm == 'proposed':
        dbscan = DBSCAN(eps=10, min_samples=10)
        dbscan_labels = dbscan.fit_predict(region_data)

        clustered_indices = defaultdict(list)
        for idx, label in enumerate(dbscan_labels):
            if label != -1:
                clustered_indices[label].append(idx)

        clusters = defaultdict(list)
        for cluster_label, indices in clustered_indices.items():
            comp_times_for_cluster = comp_times[indices].reshape(-1, 1)
            
            kmeans = KMeans(n_clusters=3, random_state=1)
            kmeans_labels = kmeans.fit_predict(comp_times_for_cluster)

            cluster_medians = []
            for k_label in range(3):
                cluster_times = comp_times_for_cluster[kmeans_labels == k_label]
                median_val = np.median(cluster_times)
                cluster_medians.append((k_label, median_val))

            sorted_clusters = sorted(cluster_medians, key=lambda x: -x[1])
            label_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(sorted_clusters)}

            reordered_labels = np.array([label_mapping[label] for label in kmeans_labels])

            for i, reordered_label in enumerate(reordered_labels):
                clusters[(cluster_label, reordered_label)].append(indices[i])
    elif args.algorithm == 'speed':
        kmeans = KMeans(n_clusters=3, random_state=1)
        kmeans.fit(comp_times)

        labels = kmeans.labels_

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

    if args.algorithm == 'proposed' or args.algorithm == 'speed':
        for key, indices in sorted(clusters.items()):
            print(f"Cluster {key}: Device Indices {indices}")


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
    for round in range(args.epochs):
        # randomly select clients
        clients = []

        if args.algorithm == 'proposed':
            for i in range(3):
                cur_client = np.random.choice(clusters[((round+i)%3,i)], size=int(n_clients/3), replace=False)
                clients.extend(cur_client)
        elif args.algorithm == 'speed':
            for cluster in clusters.values():
                cur_clients = np.random.choice(cluster, size=int(n_clients/3), replace=False)
                clients.extend(cur_client)
        else:
            random.shuffle(client_all)
            clients = client_all[:n_clients]
        
        clients = list(map(int, clients))

        print(clients)

        # assign clients to processes
        assigned_clients = []
        n_assigned_client = n_clients // n_processes
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
        log_rootpath = './log'
        if not os.path.exists(log_rootpath):
            os.makedirs(log_rootpath)
        accfile = open(log_rootpath + '/{}_{}_{}_L{}_C{}_{}_iid{}_spc{}.dat'.
                    format(args.dataset, args.model, args.epochs, args.local_ep,
                           args.frac, args.fed_strategy, args.iid, args.spc), "w")
        for acc in test_accs:
            str_ac = str(acc)
            accfile.write(str_ac)
            accfile.write('\n')
        accfile.close()
