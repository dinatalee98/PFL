import os

def main():
    algorithms = ["proposed", "client_selection", "random"]
    dataset = "mnist"
    n_clients = 40
    for algo in algorithms:
        if algo == "random":
            command = f"python main.py --algorithm={algo} --dataset={dataset} --beta=0.05 --local_ep=3"
        else:
            command = f"python main.py --algorithm={algo} --dataset={dataset} --beta=0.05 --local_ep=5"
        print(f"Executing: {command}")
        os.system(command)
    for algo in algorithms:
        if algo == "random":
            command = f"python main.py --algorithm={algo} --dataset={dataset} --fedprox --beta=0.05 --local_ep=3"
        else:
            command = f"python main.py --algorithm={algo} --dataset={dataset} --fedprox --beta=0.05 --local_ep=5"
        print(f"Executing: {command}")
        os.system(command)

if __name__ == "__main__":
    main()
