import os

def main():
    algorithms = ["proposed", "client_selection", "pipeline", "fedavg"]
    
    for algo in algorithms:
        command = f"python main.py --algorithm={algo}"
        print(f"Executing: {command}")
        os.system(command)

if __name__ == "__main__":
    main()