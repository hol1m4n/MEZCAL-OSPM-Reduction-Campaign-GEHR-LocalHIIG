import os
from pathlib import Path
import shutil
from astropy.io import fits




CARPETA = "/home/hollman/GEHR_specs/2023jun_spm/nite01/"

def id_asig(name):
    if "arc" in name.lower():
        return ['comp','ThAr','ARCS']
    elif "bias" in name.lower():
        return ['zero','Bias','BIAS']
    elif "flat" in name.lower():
        return ['flat','Flat','FLATS']
    elif "focus" in name.lower():
        return ['object','Focus']
    elif "image" in name.lower():
        return ['object','Image','IMAGES']
    else:
        if "test" in name.lower():
            return ['object','Test','OBJS']
        else:
            if 'N' in name:
                return ['object',(name.replace('N','NGC').replace('.fits','')),'OBJS']
            if 'M' in name:
                return ['object',(name.replace('M','MESSIER')).replace('.fits',''),'OBJS']
            if 'J' in name:
                return ['object',name.replace('.fits',''),'OBJS']

def correct_fits(file,folder):
    with fits.open(folder+file, mode="update") as hdul:
        header = hdul[0].header

        for key, value in SITE_INFO.items():
            if key not in header:
                header[key] = value

        imagetyp, object_name = id_asig(file)[0:2]

        if imagetyp is not None:
            header["IMAGETYP"] = imagetyp

        if object_name is not None:
            header["OBJECT"] = object_name
        hdul.flush()


def main():
    if not os.path.exists(CARPETA):
        raise FileNotFoundError(f"No existe la carpeta: {CARPETA}")

    fits_files = [f for f in os.listdir(CARPETA) if f.endswith(".fits")]
    fits_files.sort()

    print(f"Encontrados {len(fits_files)} archivos FITS.")

    for file in fits_files:
        correct_fits(file, CARPETA)

    folders = ["ARCS", "FLATS", "OBJS", "BIAS", "IMAGES", "FOCUS"]

    shutil.copy(str(os.getcwd() + '/arc_lacos.cl'), str(os.path.join(CARPETA, folders[0])))
    shutil.copy(str(os.getcwd() + '/fitcoords_sequence.cl'), str(os.path.join(CARPETA, folders[0])))
    shutil.copy(str(os.getcwd() + '/obj_lacos.cl'), str(os.path.join(CARPETA, folders[2])))



    list_generator = os.getcwd() + '/ls_gen.py'
    for folder in folders:
        dest = os.path.join(CARPETA, folder)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(str(list_generator), str(dest))


    for file in fits_files:
        fits_path = CARPETA + file
        dest_folder = CARPETA + id_asig(file)[-1] + "/"
        shutil.move(str(fits_path), str(dest_folder))


    dest_test = os.path.join(CARPETA,'OBJS/TEST/')
    os.makedirs(dest_test, exist_ok=True)

    test_files = [f for f in os.listdir(os.path.join(CARPETA,'OBJS/')) if 'test'in f.lower()]

    for file in test_files:
        fits_path = os.path.join(CARPETA,'OBJS/') + file

        shutil.move(str(fits_path), str(dest_test))

    os.makedirs(os.path.join(CARPETA, folders[0]) + '/database/', exist_ok=True)
    shutil.copy(str(os.getcwd() + '/ls_lcor.py'), str(os.path.join(CARPETA, folders[0])))

if __name__ == "__main__":
    main()
        

