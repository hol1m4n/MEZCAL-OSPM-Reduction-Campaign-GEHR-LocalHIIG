
import os
import shutil
import ast
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from joblib import Parallel, delayed
import pyfiglet


def notify(frase,mode='m'):
    if mode == 'm':
        print('\n')
        print(f'{frase}')
        print('\n')
    if mode == 'M':
        NOTIFY_font = pyfiglet.Figlet(font='colossal')
        print('\n')
        text_notify = NOTIFY_font.renderText(f'{frase} \n')
        print(text_notify)
        print('\n')



class Plotganizer:
    def __init__(self,MAIN_FOLDER=None):
        self.MAIN_FOLDER = MAIN_FOLDER
        # Coordenadas de San Pedro Martir
        self.SITE_INFO = {
            "SITELAT": 31.04417,
            "SITELONG": -115.46361,
            "SITELEV": 2800,
        }

        self.folders = ["ARCS", "FLATS", "OBJS", "BIAS", "IMAGES", "FOCUS"]

    # Identificacion del tipo de imagen con que se esta tratando

    def name_assignment(self,name=''):
        if name != '':
            if "arc" in name.lower():
                return ['comp','ThAr','ARCS']
            elif "bias" in name.lower():
                return ['zero','Bias','BIAS']
            elif "flat" in name.lower():
                return ['flat','Flat','FLATS']
            elif "focus" in name.lower() or "foco" in name.lower():
                return ['object','Focus','FOCUS']
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
                

        # Agregar informacion adicional a los headers

    def header_addition(self,file='',folder=''):
        if file != '' and folder != '':
            with fits.open(folder+file, mode="update") as hdul:
                header = hdul[0].header

                for key, value in self.SITE_INFO.items():
                    if key not in header:
                        header[key] = value

                imagetyp, object_name = self.name_assignment(file)[0:2]

                if imagetyp is not None:
                    header["IMAGETYP"] = imagetyp

                if object_name is not None:
                    header["OBJECT"] = object_name
                hdul.flush()


    # Clasificador de archivos y arreglos de header
    def executor(self):
        if not os.path.exists(self.MAIN_FOLDER):
            raise FileNotFoundError(f"No existe la carpeta: {self.MAIN_FOLDER}")
        fits_files = [f for f in os.listdir(self.MAIN_FOLDER) if f.endswith(".fits")]
        fits_files.sort()

        print(f"Encontrados {len(fits_files)} archivos FITS.")

        for file in fits_files:
            self.header_addition(file, self.MAIN_FOLDER)

        for folder in self.folders:
            dest = os.path.join(self.MAIN_FOLDER, folder)
            os.makedirs(dest, exist_ok=True)

        for file in fits_files:
            fits_path = self.MAIN_FOLDER + file
            dest_folder = self.MAIN_FOLDER + self.name_assignment(file)[-1] + "/"
            shutil.move(str(fits_path), str(dest_folder))
        
        dest_test = os.path.join(self.MAIN_FOLDER,'OBJS/TEST/')
        os.makedirs(dest_test, exist_ok=True)

        test_files = [f for f in os.listdir(os.path.join(self.MAIN_FOLDER,'OBJS/')) if 'test'in f.lower()]

        for file in test_files:
            fits_path = os.path.join(self.MAIN_FOLDER,'OBJS/') + file

            shutil.move(str(fits_path), str(dest_test))

        os.makedirs(os.path.join(self.MAIN_FOLDER, self.folders[0]) + '/database/', exist_ok=True)
        os.makedirs(os.path.join(self.MAIN_FOLDER, self.folders[2]) + '/database/', exist_ok=True)


    def image_plotter(self,path=''):
        if path != '':
            fits_file = fits.open(path)
            data = fits_file[0].data
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(1, 1, 1)
            vmin = np.percentile(data, 5)
            vmax = np.percentile(data, 95)

            if 'msk' in path:
                im = ax.imshow(data, 
                            cmap='viridis', 
                            origin='lower',
                            aspect='auto')
            else:
                im = ax.imshow(data, 
                            cmap='viridis', 
                            origin='lower',
                            vmin=vmin, 
                            vmax=vmax, 
                            aspect='auto')

            fig.savefig(path.replace('.fits', '.png'),dpi=150)
            plt.close(fig)
            fits_file.close()

    def png_x_folder(self):
        if not os.path.exists(self.MAIN_FOLDER):
            return
        folders = [f for f in os.listdir(self.MAIN_FOLDER) 
                   if os.path.isdir(os.path.join(self.MAIN_FOLDER, f))]
        folders.sort()

        for folder in folders:
            tmp_nameF = os.path.join(self.MAIN_FOLDER, folder)
            fits_to_process = [
                f for f in os.listdir(tmp_nameF) 
                if f.endswith(".fits") and not os.path.exists(os.path.join(tmp_nameF, f.replace('.fits', '.png')))
                and not '1d' in f.lower() and not '.ms' in f.lower()
            ]
            fits_to_process.sort()

            if fits_to_process:
                Parallel(n_jobs=-1)(
                    delayed(self.image_plotter)(os.path.join(tmp_nameF, f)) 
                    for f in fits_to_process
                )                           




def list_generator(FOLDER=os.getcwd(), List_in='', List_out='', out_key='',img_type=''):

    fits_files = [f for f in os.listdir(FOLDER) if f.endswith(".fits") and img_type in f.lower()]

    fits_files.sort()

    path_in = os.path.join(FOLDER, List_in)

    if not os.path.exists(path_in):
        with open(os.path.join(FOLDER, f'{List_in}'), "w") as infile:
            for file in fits_files:
                infile.write(f"{FOLDER}/{file}" + "\n")

    with open(os.path.join(FOLDER, f'{List_out}'), "w") as outfile:
        for file in fits_files:
            if not os.path.exists(path_in):
                outfile.write(f"{FOLDER}/{file.replace('.fits', '')}_{out_key}.fits" + "\n")
            else:
                tmp_name = file.replace('.fits', '')
                tmp_name = tmp_name.replace(f"_{img_type}", "")
                outfile.write(f"{FOLDER}/{tmp_name}_{out_key}.fits\n")
    
    return fits_files


def leer_o_crear_logfile(ruta):
    dict_nulo = {
        '-1': [False,' Mirror creation'],
        '0':  [False, ' Set instrument'],
        '1':  [False, ' Bias list'],
        '2':  [False, ' Bias ccdproc'],
        '3':  [False, ' Masterbias'],
        '4':  [False, ' Flat list'],
        '5':  [False, ' Flat ccdproc'],
        '6':  [False, ' Masterflat'],
        '7':  [False, ' Normalize flat'],
        '8':  [False, ' Arc list'],
        '9':  [False, ' Arc ccdproc'],
        '10': [False, ' Arc cr list'],
        '11': [False, ' Arc cosmic removal'],
        '12': [False, ' Obj list'],
        '13': [False, ' Obj ccdproc'],
        '14': [False, ' Obj cr list'],
        '15': [False, ' Obj cosmic removal'],
        '16': [False, ' WL calib'],
        '17': [False, ' FitCoords a arcos'],
        '18': [False, ' Transform a arcos'],
        '19': [False, ' Transform a objetos'],
        '20': [False, ' Busqueda de espectros calibrados y conversion a 1D'],
        '21': [False, ' Plotear espectros 1d extraidos']
    }

    if not os.path.exists(ruta):
        print(f"Archivo no encontrado. Creando {ruta} con valores iniciales...")
        with open(ruta, 'w') as f:
            import pprint
            f.write(pprint.pformat(dict_nulo, sort_dicts=False))
        return dict_nulo

    try:
        with open(ruta, 'r') as f:
            contenido = f.read().strip()
            
            if not contenido: 
                return dict_nulo
            return ast.literal_eval(contenido)

    except (SyntaxError, ValueError):
        print(f"Error de formato en {ruta}. Cargando valores por defecto.")
        return dict_nulo


def guardar_logfile(ruta, diccionario):
    with open(ruta, 'w') as f:
        from pprint import pformat
        f.write(pformat(diccionario, sort_dicts=False))




def task_checker(task,log_dict):
    current_task = task
    previous_tasks = [str(i) for i in range(-1, current_task)]
    task_verifier = all(log_dict.get(t, [False])[0] is True for t in previous_tasks)
    return task_verifier    










