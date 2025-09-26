import os

def main():
    algorithms = ["proposed", "client_selection", "random"]
    dataset = "cifar10"
    beta = 0.05
    n_clients = 50
    for algo in algorithms:
        if algo == "random":
            command = f"python main.py --algorithm={algo} --dataset={dataset} --fedprox --n_clients={n_clients} --beta={beta} --local_ep=3"
        else:
            command = f"python main.py --algorithm={algo} --dataset={dataset} --fedprox --n_clients={n_clients} --beta={beta} --local_ep=5"
        print(f"Executing: {command}")
        os.system(command)
    for algo in algorithms:
        if algo == "random":
            command = f"python main.py --algorithm={algo} --dataset={dataset} --n_clients={n_clients} --beta={beta} --local_ep=3"
        else:
            command = f"python main.py --algorithm={algo} --dataset={dataset} --n_clients={n_clients} --beta={beta} --local_ep=5"
        print(f"Executing: {command}")
        os.system(command)

if __name__ == "__main__":
    main()
