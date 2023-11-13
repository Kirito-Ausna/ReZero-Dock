import os
import sys
sys.path.append('/root/Generative-Models/ReDock/')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_path", type=str, default="results/user_predictions_testset", help="Path to a result folder that contains true_pocket.pkl")
parser.add_argument("--dst_path", type=str, default="results/esmfold_ReDockv1_apo", help="Path to a result folder that don't contains true_pocket.pkl")

args = parser.parse_args()

for name in os.listdir(args.src_path):
    src = os.path.join(args.src_path, name, "true_pockect.pkl") # there is a typo in the original code
    dst_dir = os.path.join(args.dst_path, name)
    if len(os.listdir(dst_dir)) == 0: # skip failed cases
        continue
    else:
        dst = os.path.join(dst_dir, "true_pocket.pkl")
    if os.path.exists(src) and not os.path.exists(dst):
        os.system("cp {} {}".format(src, dst))