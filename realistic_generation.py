# Highly improved generation script for large scale realistic scenerios
# Just the top script that call ReZero_Docking script automatically
# set visible devices
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,4,5,6"
from argparse import ArgumentParser
from tqdm import tqdm
import subprocess
import time
import pdb

parser = ArgumentParser()
parser.add_argument('--csv_folder', type=str, default="data/crossdock_representative_csv")
parser.add_argument('--output_folder', type=str, default="results/crossdock")
parser.add_argument('--restart_id', type=int, default=-1)

parser.add_argument('--num_agents', type=int, default=4)
parser.add_argument('--complex_per_batch', type=int, default=10)
parser.add_argument('--samples_per_complex', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for preprocessing')

parser.add_argument('--model_dir', type=str, default="workdir/ReDock_baseline")
parser.add_argument('--ckpt_path', type=str, default="best_ema_inference_epoch_model.pt")
parser.add_argument('--confidence_model_dir', type=str, default='workdir/paper_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')
parser.add_argument('--cache_path', type=str, default='experiments/cache/crossdock', help='Path to folder where the cache is stored')
parser.add_argument('--mode', type=str, default='crossdock')
args = parser.parse_args()

if args.output_folder is not None and not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

def run_docking(args, csv, device):
    target_name = os.path.basename(csv).split(".")[0]
    cmd = (
        f'python ReZero_Docking.py '
        f'--protein_ligand_csv {csv} '
        f'--out_dir {args.output_folder}/{target_name} '
        f'--model_dir {args.model_dir} '
        f'--ckpt {args.ckpt_path} '
        f'--complex_per_batch {args.complex_per_batch} '
        f'--samples_per_complex {args.samples_per_complex} '
        f'--batch_size {args.batch_size} '
        f'--num_workers {args.num_workers} '
        f'--confidence_model_dir {args.confidence_model_dir} '
        f'--confidence_ckpt {args.confidence_ckpt} '
        f'--device {device} '
        f'--mode {args.mode} '
        f'--cache_path {args.cache_path}'
    )

    log_file = f'{args.output_folder}/logs/{target_name}.log'
    # with open(log_file, 'w') as log: 
    # continue to write to the same log file
    with open(log_file, 'a') as log:
        process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    return process
   
csv_id = 0
csvs = os.listdir(args.csv_folder)[args.restart_id + 1:]

pbar = tqdm(total=len(csvs))
# initialize processes
processes = []
csv_in_processes = []
for i in range(args.num_agents):
    device = f"cuda:{i}"
    csv_path = os.path.join(args.csv_folder, csvs[csv_id])
    processes.append(run_docking(args, csv_path, device))
    csv_in_processes.append(csvs[csv_id])
    csv_id += 1
    # pbar.update(1)

# monitor the status of processes, when one is finished, start a new one
# query the status of processes every 120 seconds
try:
    while len(processes) > 0:
        deleted = []
        for i in range(len(processes)):
            if processes[i].poll() is not None:
                if csv_id < len(csvs):
                    # stop the process and start a new one
                    # processes[i].terminate()
                    # wait for the process to finish cleaning up
                    processes[i].wait()
                    print(f"Process {i} with {csv_in_processes[i]} is finished, start a new one")
                    device = f"cuda:{i}"
                    csv_path = os.path.join(args.csv_folder, csvs[csv_id])
                    processes[i] = run_docking(args, csv_path, device)
                    csv_in_processes[i] = csvs[csv_id]
                    csv_id += 1
                    pbar.update(1) # update the progress bar when a process is finished
                else: # the final num_agents processes are finished
                    # stop the process
                    # processes[i].terminate()
                    # wait for the process to finish cleaning up
                    processes[i].wait()
                    print(f"Process {i} with {csv_in_processes[i]} is finished, no more csvs to process, one last kiss")
                    pbar.update(1)
                    deleted.append(i)  # record the index of finished processes, and delete them later, otherwise the index will be out of range

        for j in deleted: # delete the finished processes
            processes.pop(j)
            csv_in_processes.pop(j)

        time.sleep(120)

except KeyboardInterrupt:
    print("KeyboardInterrupt, stop all processes")
    for i in range(len(processes)):
        processes[i].terminate()
        processes[i].wait()

print(f"All processes are finished, results are saved in {args.output_folder}.")
