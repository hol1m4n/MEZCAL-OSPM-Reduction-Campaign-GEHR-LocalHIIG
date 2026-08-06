import os
from pathlib import Path
import shutil
from astropy.io import fits
import argparse

def list_generatorLC(FOLDER=os.getcwd(), List_in='', List_out=''):

    path_in = os.path.join(FOLDER, List_in)

    if os.path.exists(path_in):
        database = FOLDER + '/database/'
        id_files = [f for f in os.listdir(database)]
        id_files.sort()

        with open(os.path.join(FOLDER, f'{List_out}'), "w") as outfile:
            for file in id_files:
                outfile.write(f"{file}" + "\n")

        path_clean = os.path.join(FOLDER, f"{List_in}woFormat")
        transform_list = os.path.join(FOLDER, f"{List_in}Transform")

        with open(path_in, "r") as infile, open(path_clean, "w") as cleanfile, open(transform_list, "w") as transfile:
            for line in infile:
                clean_line = line.strip().replace('.fits', '')
                cleanfile.write(f"{clean_line}\n")
                transfile.write(f"id{clean_line}{clean_line}\n")

        path_identify = os.path.join(FOLDER, f"{List_in}Identify")
        path_reidentify = os.path.join(FOLDER, f"{List_in}reIdentify")

        with open(path_in, "r") as infile, open(path_identify, "w") as IDfile, open(path_reidentify, "w") as reIDfile:
            lineas = infile.readlines()
            
            for n in range(len(lineas)):
                if n == 0:
                    IDfile.write(lineas[n])
                else:
                    reIDfile.write(lineas[n])

        lambda_correct = os.path.join(FOLDER, f"{List_in}LambdaCorrected")

        with open(path_in, "r") as infile, open(lambda_correct, "w") as lambdafile:
            for line in infile:
                lambda_line = line.strip().replace('crLA', 'lambda')
                lambdafile.write(f"{lambda_line}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--folder", type=str, help="Folder")
    parser.add_argument("--ls_in", type=str, help="Lista de entrada")
    parser.add_argument("--ls_out", type=str, help="Lista de salida")
    
    args = parser.parse_args()

    list_generatorLC(
        List_in=args.ls_in, 
        List_out=args.ls_out
    )












