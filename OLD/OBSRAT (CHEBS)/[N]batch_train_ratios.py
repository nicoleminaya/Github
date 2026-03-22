import subprocess
import argparse
import time

parser  = argparse.ArgumentParser()
parser.add_argument('--wds', default='anytown', type=str)
parser.add_argument('--tag', default='tuned_deterministic', type=str)
parser.add_argument('--deterministic', action  = "store_true", help    = "Setting random seed for sensor placement.")
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--epoch', default=500, type=int)
parser.add_argument('--gnn', default='cheb1', choices = ['cheb1','chebv2'], type=str)
parser.add_argument('--runs', default=4, type=int)
args = parser.parse_args()


# Configuration
runs = args.runs

print(f"Start")

ratios = [0.05, 0.1, 0.2, 0.4, 0.8]

for j in range(1, runs + 1):

    for i in ratios:
        print(f"\n------------------------------------------")
        print(f" Starting Ratio #{i}/{j}")
        print(f"------------------------------------------")
    
        if args.deterministic:
            # Python script to execute
            command = [
                "python", "[N]train_obsrat.py",  # USE other chebnev parameters
                "--epoch", str(args.epoch),
                "--adj", "binary",
                "--tag", str(args.tag),
                "--deploy", "xrandom",
                "--deterministic",
                "--wds", str(args.wds),
                "--obsrat", str(i), # iterate obsrat
                "--batch", str(args.batch),
                "--gnn", str(args.gnn)
            ]

        else:
            # Python script to execute
            command = [
                "python", "[N]train_obsrat.py",  # USE other chebnev parameters
                "--epoch", str(args.epoch),
                "--adj", "binary",
                "--tag", str(args.tag),
                "--deploy", "xrandom",
                "--wds", str(args.wds),
                "--obsrat", str(i), # iterate obsrat
                "--batch", str(args.batch),
                "--gnn", str(args.gnn)
            ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f" Error occurred in run #{i}. Stopping.")
            break
        except KeyboardInterrupt:
            print("\n Execution stopped by user.")
            break


print("\n Finish")