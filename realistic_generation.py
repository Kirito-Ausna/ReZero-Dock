# Highly improved generation script for large scale realistic scenerios
# Just the top script that call ReZero_Docking script automatically
from argparse import ArgumentParser
import os
from tqdm import tqdm
import subprocess
import time

parser = ArgumentParser()
parser.add_argument('--csv_folder', type=str, default="data/crossdock_csv")
parser.add_argument('--output_folder', type=str, default="results/crossdock")
parser.add_argument('--restart_id', type=int, default=0)

parser.add_argument('--num_agents', type=int, default=8)
parser.add_argument('--complex_per_batch', type=int, default=12)
parser.add_argument('--samples_per_complex', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for preprocessing')

parser.add_argument('--model_dir', type=str, default="workdir/paper_score_model")
parser.add_argument('--ckpt_path', type=str, default="best_ema_inference_epoch_model.pt")
parser.add_argument('--confidence_model_dir', type=str, default='workdir/paper_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')
parser.add_argument('--cache_path', type=str, default='data/example/dataset_cache', help='Path to folder where the cache is stored')
parser.add_argument('--mode', type=str, default='crossdock')
args = parser.parse_args()

if args.output_folder is not None and not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

def run_docking(args, csv, device):
    target_name = os.path.basename(csv).split(".")[0]
    cmd = f"python ReZero_Docking.py 
              --protein_ligand_csv {csv} 
              --out_dir {args.output_folder}/{target_name} 
              --complex_per_batch {args.complex_per_batch} 
              --samples_per_complex {args.samples_per_complex} 
              --batch_size {args.batch_size} 
              --num_workers {args.num_workers} 
              --model_dir {args.model_dir} 
              --ckpt {args.ckpt_path} 
              --confidence_model_dir {args.confidence_model_dir} 
              --confidence_ckpt {args.confidence_ckpt} 
              --cache_path {args.cache_path} 
              --device {device} 
              --mode {args.mode}"
    # start process with cmd and show output in the screen
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    return process
    
csv_id = 0
csvs = os.listdir(args.csv_folder)[args.restart_id + 1:]

# initialize processes
processes = []
for i in range(args.num_agents):
    device = f"cuda:{i}"
    csv_path = os.path.join(args.csv_folder, csvs[csv_id])
    processes.append(run_docking(args, csv_path, device))
    csv_id += 1

# monitor the status of processes, when one is finished, start a new one
pbar = tqdm(total=len(csvs))
# query the status of processes every 120 seconds
while len(processes) > 0:
    for i in range(len(processes)):
        if processes[i].poll() is not None:
            if csv_id < len(csvs):
                # stop the process and start a new one
                processes[i].terminate()
                print(f"Process {i} with {csv_id}:{csvs[csv_id]} is finished, start a new one")
                device = f"cuda:{i}"
                csv_path = os.path.join(args.csv_folder, csvs[csv_id])
                processes[i] = run_docking(args, csv_path, device)
                csv_id += 1
                pbar.update(1)
            else: # the final num_agents processes are finished
                # stop the process
                processes[i].terminate()
                processes.pop(i)
                pbar.update(1)
    time.sleep(120)

print(f"All processes are finished, results are saved in {args.output_folder}.")
