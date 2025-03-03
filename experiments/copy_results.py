import os
from shutil import copyfile
result_dir = "results/IL17-9FL3"
des_dir = "data/IL17_2pockets/IL17/04 Protein-Molecule Flexible Docking/9FL3"

# copy all files in result_dir to des_dir, keep the directory structure and only accept files named after rank1
for root, dirs, files in os.walk(result_dir):
    for file in files:
        if file == "rank1.sdf" or file == "rank1_pocket.pdb":
            src_file = os.path.join(root, file)
            des_file = os.path.join(des_dir, os.path.relpath(src_file, result_dir))
            os.makedirs(os.path.dirname(des_file), exist_ok=True)
            # os.system("cp {} {}".format(src_file, des_file))
            # using python copyfile instead of os.system
            copyfile(src_file, des_file)

