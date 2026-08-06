import os
from pathlib import Path
import shutil
from astropy.io import fits
import argparse

def list_generator(FOLDER=os.getcwd(), List_in='', List_out='', out_key='',img_type=''):

    fits_files = [f for f in os.listdir(FOLDER) if f.endswith(".fits") and img_type in f.lower()]

    fits_files.sort()

    path_in = os.path.join(FOLDER, List_in)

    if not os.path.exists(path_in):
        with open(os.path.join(FOLDER, f'{List_in}'), "w") as infile:
            for file in fits_files:
                infile.write(file + "\n")

    with open(os.path.join(FOLDER, f'{List_out}'), "w") as outfile:
        for file in fits_files:
            if not os.path.exists(path_in):
                outfile.write(f"{file.replace('.fits', '')}_{out_key}.fits" + "\n")
            else:
                tmp_name = file.replace('.fits', '')
                tmp_name = tmp_name.replace(f"_{img_type}", "")
                outfile.write(f"{tmp_name}_{out_key}.fits\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--folder", type=str, help="Folder")
    parser.add_argument("--ls_in", type=str, help="Lista de entrada")
    parser.add_argument("--ls_out", type=str, help="Lista de salida")
    parser.add_argument("--o_key", type=str, help="Clave de salida")
    parser.add_argument("--im_ty", type=str, help="Tipo de imagen a buscar")
    
    args = parser.parse_args()

    list_generator(
        List_in=args.ls_in, 
        List_out=args.ls_out, 
        out_key=args.o_key,
        img_type=args.im_ty
    )





