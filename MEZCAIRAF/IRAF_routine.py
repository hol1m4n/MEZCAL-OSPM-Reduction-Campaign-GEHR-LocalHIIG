# Librerias esenciales

import os
from pathlib import Path
import shutil
from astropy.io import fits
from lacosmic import remove_cosmics
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
#import pyraf
from pyraf import iraf
from pyraf import gki
import ast
from pprint import pprint
import random
from scipy.ndimage import median_filter
import matplotlib.gridspec as gridspec
import glob
import pyfiglet

NOTIFY_font = pyfiglet.Figlet(font='colossal')


text_notify = NOTIFY_font.renderText('_____')

text_notify2 = NOTIFY_font.renderText('START \n')




print(text_notify)
print(text_notify2)


def notify(frase):
    print('\n')
    print(f'{frase}')
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
        
        #self.executor()

    # Identificacion del tipo de imagen con que se esta tratando

    def name_assignment(self,name=''):
        if name != '':
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
            #print(path)
            fits_file = fits.open(path)
            data = fits_file[0].data
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(1, 1, 1)
            # 1. Usamos cmap='Greys' para escala de grises (o 'gist_yarg' para el inverso)
            # 2. Ajustamos vmin y vmax para "estirar" el contraste y que se vea claro
            vmin = np.percentile(data, 5)   # Valor mínimo (ajusta el fondo)
            vmax = np.percentile(data, 95)  # Valor máximo (ajusta el brillo de las líneas)

            if 'msk' in path:
                im = ax.imshow(data, 
                            cmap='viridis', 
                            origin='lower',
                            aspect='auto') # 'auto' ayuda si la imagen se ve muy estirada
            else:
                im = ax.imshow(data, 
                            cmap='viridis', 
                            origin='lower',
                            vmin=vmin, 
                            vmax=vmax, 
                            aspect='auto') # 'auto' ayuda si la imagen se ve muy estirada

                # Guardamos la imagen
            fig.savefig(path.replace('.fits', '.png'),dpi=150)
            
            # CRUCIAL: Cerramos la figura para que no se renderice en el notebook
            plt.close(fig)
            
            # Cerramos el archivo fits para liberar memoria
            fits_file.close()

    def png_x_folder(self):
        if not os.path.exists(self.MAIN_FOLDER):
            return

        # 1. Obtener solo carpetas reales
        folders = [f for f in os.listdir(self.MAIN_FOLDER) 
                   if os.path.isdir(os.path.join(self.MAIN_FOLDER, f))]
        folders.sort()

        for folder in folders:
            tmp_nameF = os.path.join(self.MAIN_FOLDER, folder)

            #print(tmp_nameF)
            
            # 2. Archivos FITS que NO tienen su PNG listo
            fits_to_process = [
                f for f in os.listdir(tmp_nameF) 
                if f.endswith(".fits") and not os.path.exists(os.path.join(tmp_nameF, f.replace('.fits', '.png')))
                and not '1d' in f.lower() and not 'ms' in f.lower()
            ]
            fits_to_process.sort()

            if fits_to_process:
                # 3. Paralelismo directo sobre la función de dibujo
                # Pasamos la ruta completa del FITS a procesar
                Parallel(n_jobs=-1)(
                    delayed(self.image_plotter)(os.path.join(tmp_nameF, f)) 
                    for f in fits_to_process
                )




work_dir = '/home/hollman/GEHR_specs/2023jun_spm/nite01/'


notify('Organizacion de archivos y asignacion de headers')



Plotganizer(work_dir).executor()


notify('Conversion de imagenes crudas .fits a .png')


Plotganizer(work_dir).png_x_folder()

#'''

notify('Cargando librerias esenciales')

iraf.noao()
iraf.imred()
iraf.ccdred()



#'''



# Creador de listas

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
    # 1. Definimos los valores por defecto
    dict_nulo = {
        '0': [False, ' Set instrument'],
        '1': [False, ' Bias list'],
        '2': [False, ' Bias ccdproc'],
        '3': [False, ' Masterbias'],
        '4': [False, ' Flat list'],
        '5': [False, ' Flat ccdproc'],
        '6': [False, ' Masterflat'],
        '7': [False, ' Normalize flat'],
        '8': [False, ' Arc list'],
        '9': [False, ' Arc ccdproc'],
        '10': [False, ' Arc cr list'],
        '11': [False, ' Arc cosmic removal'],
        '12': [False, ' Obj list'],
        '13': [False, ' Obj ccdproc'],
        '14': [False, ' Obj cr list'],
        '15': [False, ' Obj cosmic removal'],
        '16': [True, ' WL calib'],
        '17': [False, ' FitCoords a arcos'],
        '18': [False, ' Transform a arcos'],
        '19': [False, ' Transform a objetos'],
        '20': [False, ' Busqueda de espectros calibrados y conversion a 1D'],
        '21': [False, ' Plotear espectros 1d extraidos']
    }

    if not os.path.exists(ruta):
        # 3. Si el archivo no existe, lo creamos con saltos de línea
        print(f"Archivo no encontrado. Creando {ruta} con valores iniciales...")
        with open(ruta, 'w') as f:
            # Usamos pformat para convertir el dict a un string bonito con saltos de línea
            import pprint
            f.write(pprint.pformat(dict_nulo, sort_dicts=False))
        return dict_nulo

    try:
        # 2. Intentamos abrir el archivo para leer
        with open(ruta, 'r') as f:
            contenido = f.read().strip()
            
            if not contenido: 
                return dict_nulo
            
            # ast.literal_eval ignora los saltos de línea, así que leerá el dict normal
            return ast.literal_eval(contenido)

    except (SyntaxError, ValueError):
        # Por si el archivo tiene errores de formato
        print(f"Error de formato en {ruta}. Cargando valores por defecto.")
        return dict_nulo

def guardar_logfile(ruta, diccionario):
    with open(ruta, 'w') as f:
        from pprint import pformat
        # sort_dicts=False mantiene el orden del 1 al N
        f.write(pformat(diccionario, sort_dicts=False))






RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')







notify('Fijar el instrumento para CCDPROC')

if RS['0'][0] == True:
    notify('Ya esta cargado el instrumento')
else:
    notify('Fijar el instrumento y salir')
    iraf.setinstrument(instrument = 'mezcal' ,site = 'ospm' ,directory = 'ccddb$')
    RS['0'][0] = True





















notify('Comienza creacion del MASTERFLAT')



notify('Generacion de listas de bias')

# Creacion de la lista para los bias

if RS['1'][0] == False:
    list_generator(FOLDER=f'{work_dir}BIAS', 
                List_in='Bias_raw', 
                List_out='Bias_trim', 
                out_key='trim',
                img_type='bias')
    RS['1'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)




# Corte de las imagenes

iraf.noao()
iraf.imred()
iraf.ccdred()



if RS['2'][0] == True:
    notify('Los bias ya estan cortados con CCDPROC')
else:
    notify('Cortando bias con CCDPROC')
    iraf.ccdproc(
        images = str(f'@{work_dir}BIAS/Bias_raw'),
        output = str(f'@{work_dir}BIAS/Bias_trim'),
        ccdtype = "zero",
        max_cache = 0,
        noproc = "no",
        fixpix = "no",
        overscan = "no",
        trim = "yes",
        zerocor = "no",
        darkcor = "no",
        flatcor = "no",
        illumcor = "no",
        fringecor = "no",
        readcor = "no",
        scancor = "no",
        readaxis = "line",
        trimsec = "[10:1024,86:1024]",
        minreplace = 1.0,
        scantype = "shortscan",
        nscan = 1,
        interactive = "no",
        function = "legendre",
        order = 2,
        sample = "*",
        naverage = 1,
        niterate = 1,
        low_reject = 3.0,
        high_reject = 3.0,
        grow = 0.0
    )

    Plotganizer(work_dir).png_x_folder()
    RS['2'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)



if RS['3'][0] == True:
    notify('El masterbias existe')
else:
    notify('Creando el masterbias')
    iraf.zerocombine( 
        input = f'@{work_dir}BIAS/Bias_trim',
        output = f"{work_dir}BIAS/masterbias.fits",
        combine = "average",
        reject = "minmax",
        ccdtype = "zero",
        process = "no",
        delete = "no",
        clobber = "no",
        scale = "none",
        nlow = 0,
        nhigh = 1,
        nkeep = 1,
        mclip = "yes",
        lsigma = 3.0,
        hsigma = 3.0,
        rdnoise = 0.0,
        gain = 1.0,
        snoise = 0.0,
        pclip = -0.5,
        blank = 0.0
    )
    Plotganizer(work_dir).png_x_folder()
    RS['3'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)








# ─────────────────────────────────────────────
# PASO 3 — MASTERBIAS
# Qué buscar: perfil plano, histograma angosto, sin gradientes
# ─────────────────────────────────────────────
def diag_masterbias(work_dir):
    path = f'{work_dir}BIAS/masterbias.fits'
    with fits.open(path) as hdul:
        data = hdul[0].data

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Diagnóstico Masterbias', fontsize=14)

    # 2D
    ax = axes[0, 0]
    vmin, vmax = np.percentile(data, [1, 99])
    im = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title('Imagen 2D')

    # Perfil espacial
    ax = axes[0, 1]
    ax.plot(np.median(data, axis=1), lw=0.8)
    ax.axhline(np.median(data), color='r', linestyle='--', label=f'Mediana global = {np.median(data):.2f}')
    ax.set_xlabel('Fila Y')
    ax.set_ylabel('ADUs')
    ax.set_title('Perfil espacial  [debe ser plano]')
    ax.legend()

    # Perfil espectral
    ax = axes[1, 0]
    ax.plot(np.median(data, axis=0), lw=0.8)
    ax.axhline(np.median(data), color='r', linestyle='--')
    ax.set_xlabel('Columna X')
    ax.set_ylabel('ADUs')
    ax.set_title('Perfil espectral  [debe ser plano]')

    # Histograma
    ax = axes[1, 1]
    ax.hist(data.flatten(), bins=200, color='steelblue', edgecolor='none')
    ax.set_xlabel('ADUs')
    ax.set_ylabel('N píxeles')
    mu, sigma = np.mean(data), np.std(data)
    ax.set_title(f'Histograma  μ={mu:.2f}  σ={sigma:.2f} ADUs\n'
                 f'[σ debe ser ≈ readnoise/√N_bias]')

    plt.tight_layout()
    plt.savefig(f'{work_dir}BIAS/diag_masterbias.png', dpi=150)
    plt.close()
    print(f'  → Masterbias: μ={mu:.2f}, σ={sigma:.3f} ADUs')



#diag_masterbias(work_dir)


































notify('Generacion de listas de flats')

# Creacion de la lista para los flats

if RS['4'][0] == False:
    list_generator(FOLDER=f'{work_dir}FLATS', 
                List_in='Flat_raw', 
                List_out='Flat_trim', 
                out_key='trim',
                img_type='flat')
    RS['4'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)






if RS['5'][0] == True:
    notify('Los flats ya estan cortados con CCDPROC + masterbias')
else:
    notify('Cortando flats con CCDPROC + masterbias')
    iraf.ccdproc(
        images = f'@{work_dir}FLATS/Flat_raw',
        output = f'@{work_dir}FLATS/Flat_trim',
        ccdtype = "flat",
        max_cache = 0,
        noproc = "no",
        fixpix = "no",
        overscan = "no",
        trim = "yes",
        zerocor = "yes",
        darkcor = "no",
        flatcor = "no",
        illumcor = "no",
        fringecor = "no",
        readcor = "no",
        scancor = "no",
        readaxis = "line",
        trimsec = "[10:1024,86:1024]",
        zero = f"{work_dir}BIAS/masterbias.fits",
        minreplace = 1.0,
        scantype = "shortscan",
        nscan = 1,
        interactive = "no",
        function = "legendre",
        order = 2,
        sample = "*",
        naverage = 1,
        niterate = 1,
        low_reject = 3.0,
        high_reject = 3.0,
        grow = 0.0
    )

    Plotganizer(work_dir).png_x_folder()
    RS['5'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)



if RS['6'][0] == True:
    notify('El masterflat existe')
else:
    notify('Creando el masterflat')
    iraf.flatcombine(
        input = f'@{work_dir}FLATS/Flat_trim',
        output = f'{work_dir}FLATS/masterflat.fits',
        combine = "median",
        reject = "avsigclip",
        ccdtype = "flat",
        process = "no",
        subsets = "no",
        delete = "no",
        clobber = "no",
        scale = "mode",
        nlow = 3,
        nhigh = 3, 
        nkeep = 10,
        mclip = "yes", 
        lsigma = 2.5, 
        hsigma = 2.5, 
        rdnoise = "RDNOISE", 
        gain = "GAIN", 
        snoise = "0.0", 
        pclip = -0.5, 
        blank = 1.0
    )
    Plotganizer(work_dir).png_x_folder()
    RS['6'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)


notify('De aqui en adelante comienzan las tareas que requieren algun tipo de interaccion')

iraf.noao()
iraf.twodspec()
iraf.longslit()

iraf.set(stdgraph="xgterm")




if RS['7'][0] == True:
    notify('El masterflat normalizado existe')
else:

    notify('Enmascarando la fuga de luz... A ver si se mejora ese aspecto.')

    with fits.open(f'{work_dir}FLATS/masterflat.fits') as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header

    # Identifica la columna de la franja (inspeccionala con imarith/imexamine)
    # Por ejemplo, si la franja está en columnas 100-115:
    fuga_mask = np.zeros(data.shape, dtype=bool)
    fuga_mask[:,275:332] = True  # ajusta los índices

    # Reemplaza con mediana local suavizada
    data_smooth = median_filter(data, size=(1, 200))  # suaviza en dispersion
    data[fuga_mask] = data_smooth[fuga_mask]

    fits.writeto(f'{work_dir}FLATS/masterflat_masked.fits', data, header, overwrite=True)




    notify('Creando el masterflat normalizado previo a la correccion por iluminacion. Se requiere intervencion manual')
    iraf.response( 
        calibration = f'{work_dir}FLATS/masterflat_masked.fits',
        normalization = f'{work_dir}FLATS/masterflat_masked.fits',
        response = f'{work_dir}FLATS/FlatNpre_lumcor.fits', #f'{work_dir}FLATS/FlatNpre_lumcor.fits', f'{work_dir}FLATS/FlatNmaster.fits'
        interactive = "yes",
        threshold = "INDEF",
        sample = "*",
        naverage = 1,
        function = "spline3",
        order = 12, # Esto habria que probarlo hasta con 1000 a ver si se arregla lo de la fuga de luz
        low_reject = 3.0,
        high_reject = 3.0,
        niterate = 1,
        grow = 0.0,
        graphics = "stdgraph"
    )
    

    iraf.illumination(
        images        = f'{work_dir}FLATS/FlatNpre_lumcor.fits',
        illuminations = f'{work_dir}FLATS/FlatIllum.fits',
        interactive   = "no",
        nbins         = 10,      # divide la imagen en 10 bins espectrales para samplear
        sample        = "*",
        naverage      = -10,     # mediana de 50 lineas por bin (robusto)
        function      = "spline3",
        order         = 4,       # orden suficiente para seguir el arco
        low_reject    = 3.0,
        high_reject   = 3.0,
        niterate      = 2,
        grow          = 0.0,
        interpolator = "poly3",
        graphics      = "stdgraph"
    )


    iraf.imarith(
        operand1 = f'{work_dir}FLATS/FlatNpre_lumcor.fits',
        op       = '/',
        operand2 = f'{work_dir}FLATS/FlatIllum.fits',
        result   = f'{work_dir}FLATS/FlatNmaster.fits'
    )

    with fits.open(f'{work_dir}FLATS/FlatNmaster.fits') as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header

    global_median = np.median(data)
    print(f'Mediana global del flat: {global_median:.4f}')  # debería ser ~1.09

    fits.writeto(f'{work_dir}FLATS/FlatNmaster.fits', 
                data / global_median, header, overwrite=True)
    
    
    
    '''
    with fits.open(f'{work_dir}FLATS/FlatNmaster.fits') as hdul:
        flat = hdul[0].data
    mediana_for_bpx = np.median(flat, axis=1)

    bad_pixs = np.where((mediana_for_bpx>=1.015) | (mediana_for_bpx<=0.985))[0]

    # Crear un archivo de regiones malas para fixpix
    # Formato: columna_ini columna_fin fila_ini fila_fin
    with open(f'{work_dir}FLATS/bad_rows.dat', 'w') as f:

        for l in bad_pixs:
            f.write(f"1 1024 {l} {l}\n")


    iraf.fixpix(
        images = f'{work_dir}FLATS/FlatNmaster.fits',
        masks  = f'{work_dir}FLATS/bad_rows.dat',
        linterp = "INDEF",
        cinterp = "INDEF"
    )
    '''
    

    Plotganizer(work_dir).png_x_folder()
    RS['7'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)





# ─────────────────────────────────────────────
# PASO 6+7 — MASTERFLAT (crudo y normalizado)
# Qué buscar: perfil espacial plano en FlatNmaster (±2-3%)
# ─────────────────────────────────────────────
def diag_flats(work_dir):
    archivos = {
        'masterflat (crudo)':       f'{work_dir}FLATS/masterflat.fits',
        'FlatNpre_lumcor':           f'{work_dir}FLATS/FlatNpre_lumcor.fits',
        'FlatIllum':                 f'{work_dir}FLATS/FlatIllum.fits',
        'FlatNmaster (final)':       f'{work_dir}FLATS/FlatNmaster.fits',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Perfil espacial — cadena de flats', fontsize=14)

    for ax, (label, path) in zip(axes.flatten(), archivos.items()):
        if not Path(path).exists():
            ax.set_title(f'{label}\n[no encontrado]')
            continue
        with fits.open(path) as hdul:
            data = hdul[0].data
        med = np.median(data, axis=1)
        ax.plot(med, lw=0.8)
        ax.axhline(np.median(med), color='r', linestyle='--',
                   label=f'mediana={np.median(med):.3f}')
        ax.set_xlabel('Fila Y')
        ax.set_ylabel('Mediana por fila')
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{work_dir}FLATS/diag_flats_perfiles.png', dpi=150)
    plt.close()

    # Histograma del FlatNmaster final
    path_fn = f'{work_dir}FLATS/FlatNmaster.fits'
    if Path(path_fn).exists():
        with fits.open(path_fn) as hdul:
            data = hdul[0].data
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('FlatNmaster final — verificación', fontsize=13)

        axes[0].hist(data.flatten(), bins=300, color='steelblue', edgecolor='none')
        axes[0].axvline(1.0, color='r', linestyle='--', label='Ideal = 1.0')
        axes[0].set_xlabel('Valor del píxel')
        axes[0].set_ylabel('N píxeles')
        axes[0].set_title(f'Histograma  μ={np.mean(data):.4f}  σ={np.std(data):.4f}')
        axes[0].legend()

        axes[1].plot(np.median(data, axis=0), lw=0.8)
        axes[1].axhline(1.0, color='r', linestyle='--')
        axes[1].set_xlabel('Columna X (dirección espectral)')
        axes[1].set_ylabel('Mediana por columna')
        axes[1].set_title('Perfil espectral [debe ser ~1.0]')

        plt.tight_layout()
        plt.savefig(f'{work_dir}FLATS/diag_FlatNmaster_final.png', dpi=150)
        plt.close()


#diag_flats(work_dir)










































def cosmicos_cleaner(line):
    line = line.strip() # Reemplaza el replace('\n','')
    cr_file = line.replace('trim', 'crLA')
    msk_file = line.replace('trim', 'mskLA')

    # Abrimos el original para leer datos y header
    with fits.open(line) as hdul_orig:
        data_ = hdul_orig[0].data
        header_ = hdul_orig[0].header

        notify(f'Removiendo cosmicos de {line}')

        # Ejecución del algoritmo
        if 'arc' in line.lower():
            clean_img, cr_mask = remove_cosmics(data=data_, 
                                                effective_gain=1.0, 
                                                readnoise=5.0, 
                                                maxiter=4, 
                                                cr_threshold=4.5, 
                                                neighbor_threshold=0.3, 
                                                contrast=7.0, 
                                                border_mode='mirror')
        else:
            clean_img, cr_mask = remove_cosmics(data=data_, 
                                                effective_gain=1.0, 
                                                readnoise=5.0, 
                                                maxiter=6, 
                                                cr_threshold=3.5, 
                                                neighbor_threshold=0.25, 
                                                contrast=7.0, 
                                                border_mode='mirror')

        # 1. Guardar imagen limpia (crLA)
        # Creamos un PrimaryHDU nuevo con la data limpia y el header original
        hdu_clean = fits.PrimaryHDU(data=clean_img, header=header_)
        hdu_clean.writeto(cr_file, overwrite=True)

        # 2. Guardar máscara (mskLA)
        # Convertimos máscara a entero (int16 es suficiente y ahorra espacio)
        hdu_mask = fits.PrimaryHDU(data=cr_mask.astype('int16'), header=header_)
        hdu_mask.writeto(msk_file, overwrite=True) 





















notify('Comienza el corte de los Arcos ')



notify('Generacion de listas de arcos')

# Creacion de la lista para los bias
if RS['8'][0] == False:
    list_generator(FOLDER=f'{work_dir}ARCS', 
                List_in='Arcs_raw', 
                List_out='Arcs_trim', 
                out_key='trim',
                img_type='arc')
    RS['8'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)





if RS['9'][0] == True:
    notify('Los arcos ya estan cortados con CCDPROC + masterbias + FlatNmaster.fits')
else:
    notify('Cortando arcos con CCDPROC + masterbias + FlatNmaster.fits')
    iraf.ccdproc(
        images = f'@{work_dir}ARCS/Arcs_raw',
        output = f'@{work_dir}ARCS/Arcs_trim',
        ccdtype = "comp",
        max_cache = 0,
        noproc = "no",
        fixpix = "no",
        overscan = "no",
        trim = "yes",
        zerocor = "yes",
        darkcor = "no",
        flatcor = "yes",
        illumcor = "no", #no
        fringecor = "no",
        readcor = "no",
        scancor = "no",
        readaxis = "line",
        trimsec = "[10:1024,86:1024]",
        zero = f"{work_dir}BIAS/masterbias.fits",
        flat = f"{work_dir}FLATS/FlatNmaster.fits",
        #illum = f"{work_dir}FLATS/FlatIllum.fits",
        minreplace = 1.0,
        scantype = "shortscan",
        nscan = 1,
        interactive = "no",
        function = "legendre",
        order = 2,
        sample = "*",
        naverage = 1,
        niterate = 1,
        low_reject = 3.0,
        high_reject = 3.0,
        grow = 0.0
    )

    Plotganizer(work_dir).png_x_folder()
    RS['9'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)



notify('Comienza la remocion de cosmicos de los Arcos ')



notify('Generacion de listas de arcos (sin cosmicos)')

# Creacion de la lista para los bias


if RS['10'][0] == False:
    list_generator(FOLDER=f'{work_dir}ARCS', 
                List_in='Arcs_raw', 
                List_out='Arcs_cr', 
                out_key='crLa',
                img_type='trim')
    RS['10'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)




if RS['11'][0] == True:
    notify('Ya se removieron los cosmicos de los arcos')
else:
    notify('Empieza el bucle para remover cosmicos de los arcos...')
    path_in = f'{work_dir}ARCS/Arcs_trim'
    if os.path.exists(path_in):
        with open(path_in, "r") as infile:
            if infile:
                Parallel(n_jobs=-1)(
                    delayed(cosmicos_cleaner)(line) 
                    for line in infile
                )                  

    Plotganizer(work_dir).png_x_folder()
    RS['11'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)














notify('Comienza el corte de los Objetos ')



notify('Generacion de listas de Objetos')

# Creacion de la lista para los bias

if RS['12'][0] == False:
    list_generator(FOLDER=f'{work_dir}OBJS', 
                List_in='Obj_raw', 
                List_out='Obj_trim', 
                out_key='trim',
                img_type='o')
    RS['12'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)


if RS['13'][0] == True:
    notify('Los objetos ya estan cortados con CCDPROC + masterbias + FlatNmaster.fits')
else:
    notify('Cortando objetos con CCDPROC + masterbias + FlatNmaster.fits')
    iraf.ccdproc(
        images = f'@{work_dir}OBJS/Obj_raw',
        output = f'@{work_dir}OBJS/Obj_trim',
        ccdtype = "object",
        max_cache = 0,
        noproc = "no",
        fixpix = "no",
        overscan = "no",
        trim = "yes",
        zerocor = "yes",
        darkcor = "no",
        flatcor = "yes",
        illumcor = "no", #no
        fringecor = "no",
        readcor = "no",
        scancor = "no",
        readaxis = "line",
        trimsec = "[10:1024,86:1024]",
        zero = f"{work_dir}BIAS/masterbias.fits",
        flat = f"{work_dir}FLATS/FlatNmaster.fits",
        #illum = f"{work_dir}FLATS/FlatIllum.fits",
        minreplace = 1.0,
        scantype = "shortscan",
        nscan = 1,
        interactive = "no",
        function = "legendre",
        order = 2,
        sample = "*",
        naverage = 1,
        niterate = 1,
        low_reject = 3.0,
        high_reject = 3.0,
        grow = 0.0
    )

    Plotganizer(work_dir).png_x_folder()
    RS['13'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)


if RS['14'][0] == False:
    list_generator(FOLDER=f'{work_dir}OBJS', 
                List_in='Obj_raw', 
                List_out='Obj_cr', 
                out_key='crLa',
                img_type='trim')
    RS['14'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)




if RS['15'][0] == True:
    notify('Ya se removieron los cosmicos de los arcos')
else:
    notify('Empieza el bucle para remover cosmicos de los objetos...')
    path_in = f'{work_dir}OBJS/Obj_trim'
    if os.path.exists(path_in):
        with open(path_in, "r") as infile:
            if infile:
                Parallel(n_jobs=-1)(
                    delayed(cosmicos_cleaner)(line) 
                    for line in infile
                )                

    Plotganizer(work_dir).png_x_folder()
    RS['15'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)







# ─────────────────────────────────────────────
# PASOS 11 y 15 — REMOCIÓN DE CÓSMICOS (arcos y objetos)
# Qué buscar: fracción de píxeles enmascarados (<1% arcos, <0.5% objetos)
#             Que líneas de arco NO estén enmascaradas
# ─────────────────────────────────────────────
def diag_cosmic_removal(work_dir, tipo='ARCS', sufijo_trim='trim', sufijo_cr='crLA', sufijo_msk='mskLA'):
    folder = f'{work_dir}{tipo}/'
    trim_files = sorted(glob.glob(f'{folder}*{sufijo_trim}*.fits'))

    if not trim_files:
        print(f'  No se encontraron archivos {sufijo_trim} en {folder}')
        return

    def plotter_x_cosmic_removal_diagnostic(Cd,trim_files):
        f_trim = trim_files[Cd]
        f_cr   = f_trim.replace(sufijo_trim, sufijo_cr)
        f_msk  = f_trim.replace(sufijo_trim, sufijo_msk)

        if not Path(f_cr).exists() or not Path(f_msk).exists():
            print(f'  Archivos crLA/mskLA no encontrados para {f_trim}')
            return

        with fits.open(f_trim) as h: orig = h[0].data
        with fits.open(f_cr)   as h: clean = h[0].data
        with fits.open(f_msk)  as h: mask = h[0].data

        frac_cr = mask.sum() / mask.size * 100
        nombre = Path(f_trim).stem

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Remoción de cósmicos — {tipo} — {nombre}', fontsize=13)

        vmin, vmax = np.percentile(orig, [1, 99])
        axes[0].imshow(orig,  cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title('Original (trim)')

        axes[1].imshow(clean, cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title('Limpia (crLA)')

        axes[2].imshow(mask,  cmap='Reds',    origin='lower', aspect='auto')
        axes[2].set_title(f'Máscara CR\n{frac_cr:.3f}% píxeles afectados\n'
                        f'[bueno si <1% arcos, <0.5% objetos]')

        plt.tight_layout()
        outname = f'{folder}diag_cosmics_{tipo.lower()}_{nombre}.png'
        plt.savefig(outname, dpi=150)
        plt.close()
        print(f'  → {tipo} CR fraction: {frac_cr:.3f}%  ({"OK" if frac_cr < 1.0 else "REVISAR"})')

    
    Parallel(n_jobs=-1)(
        delayed(plotter_x_cosmic_removal_diagnostic)(Cd,trim_files) 
        for Cd in range(len(trim_files))
    )    

    '''
    for Cd in range(len(trim_files)):
    # Toma el primero como ejemplo
        f_trim = trim_files[Cd]
        f_cr   = f_trim.replace(sufijo_trim, sufijo_cr)
        f_msk  = f_trim.replace(sufijo_trim, sufijo_msk)

        if not Path(f_cr).exists() or not Path(f_msk).exists():
            print(f'  Archivos crLA/mskLA no encontrados para {f_trim}')
            return

        with fits.open(f_trim) as h: orig = h[0].data
        with fits.open(f_cr)   as h: clean = h[0].data
        with fits.open(f_msk)  as h: mask = h[0].data

        frac_cr = mask.sum() / mask.size * 100
        nombre = Path(f_trim).stem

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Remoción de cósmicos — {tipo} — {nombre}', fontsize=13)

        vmin, vmax = np.percentile(orig, [1, 99])
        axes[0].imshow(orig,  cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title('Original (trim)')

        axes[1].imshow(clean, cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title('Limpia (crLA)')

        axes[2].imshow(mask,  cmap='Reds',    origin='lower', aspect='auto')
        axes[2].set_title(f'Máscara CR\n{frac_cr:.3f}% píxeles afectados\n'
                        f'[bueno si <1% arcos, <0.5% objetos]')

        plt.tight_layout()
        outname = f'{folder}diag_cosmics_{tipo.lower()}_{nombre}.png'
        plt.savefig(outname, dpi=150)
        plt.close()
        print(f'  → {tipo} CR fraction: {frac_cr:.3f}%  ({"OK" if frac_cr < 1.0 else "REVISAR"})')
        '''



#diag_cosmic_removal(work_dir, tipo='ARCS', sufijo_trim='trim',
#                    sufijo_cr='crLA', sufijo_msk='mskLA')


#diag_cosmic_removal(work_dir, tipo='OBJS', sufijo_trim='trim',
#                    sufijo_cr='crLA', sufijo_msk='mskLA')









































iraf.noao()
iraf.twodspec()
iraf.longslit()




notify('Comienza el identify basado en 4 arcos aleatorios. Se hace el identify para cada uno de ellos y se propaga a los subsiguientes.')


if RS['16'][0] == True:
    notify('Ya se realizo la calibracion en longitud de onda, revisar el database')
else:

    original_dir = os.getcwd()
    arcs_dir = os.path.abspath(f'{work_dir}ARCS')
    os.chdir(arcs_dir)


    arc_files = [f for f in os.listdir(f'{work_dir}ARCS') if f.endswith(".fits") and 'cr' in f.lower()]
    arc_files.sort()

    integ_ = 3
    grupos = (len(arc_files) + integ_ - 1) // integ_ # tres el numero de integrantes por grupo

    for G in range(grupos):
        grupo_n = []
        for i in range(integ_):
            grupo_n.append(G * integ_ + i)
        chosen = random.choice(grupo_n)
        grupo_n.remove(chosen)
        print(grupo_n, chosen)
        with open(f'Wl_calG{G+1}', "w") as wave_calibra_group: #{work_dir}ARCS/
            for i in grupo_n:
                wave_calibra_group.write(f'{arc_files[i]} \n')  #{work_dir}ARCS/

        iraf.identify(
            images = f'{arc_files[chosen]}', #{work_dir}ARCS/
            section = "middle line",
            database = "database",
            coordlist = "linelists$thar.dat",
            nsum = 10,
            match = 10,
            maxfeatures = 150,
            zwidth = 100.0,
            ftype = "emission",
            fwidth = 5.0,
            cradius = 3.0,
            threshold = 0.0,
            minsep = 1.0,
            function = "spline3",
            order = 1,
            sample = "*",
            niterate = 1,
            low_reject = 3.0,
            high_reject = 3.0,
            grow = 0.0,
            autowrite = "yes",
            graphics = "stdgraph"
        )

        iraf.reidentify(reference = f'{arc_files[chosen]}', # {work_dir}ARCS/
            images = f'@Wl_calG{G+1}', #{work_dir}ARCS/
            interactive = "no",
            section = "middle line",
            newaps = "yes",
            override = "yes",
            refit = "yes",
            trace = "no",
            step = 10,
            nsum = 10,
            shift = 0.0,
            search = 0.0,
            nlost = 0,
            cradius = 5.0,
            threshold = 0.0,
            addfeatures = "no",
            coordlist = "linelists$thar.dat",
            match = -3.0,
            maxfeatures = 150,
            minsep = 1.0,
            database = f"database",
            logfiles = f"logfile",
            verbose = "yes",
            graphics = "stdgraph"
        )

    os.chdir(original_dir)
    RS['16'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)



# ─────────────────────────────────────────────
# PASO 16 — CALIBRACIÓN EN LONGITUD DE ONDA
# Qué buscar: RMS < 0.3 Ang, ≥ 20 líneas identificadas por arco
# ─────────────────────────────────────────────
def diag_wl_calibration(work_dir):
    database_dir = f'{work_dir}ARCS/database/'
    db_files = sorted(glob.glob(f'{database_dir}id*'))

    if not db_files:
        print('  No se encontraron archivos en database/')
        return

    resultados = []
    for db_file in db_files:
        nombre = Path(db_file).name
        rms, nlines, function, order = None, None, None, None
        pixels, wavelengths = [], []

        with open(db_file, 'r') as f:
            lines = f.readlines()

        in_features = False
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('rms'):
                try: rms = float(line.split()[1])
                except: pass
            if line.startswith('features'):
                try: nlines = int(line.split()[1])
                except: pass
            if line.startswith('function'):
                function = line.split()[1]
            if line.startswith('order'):
                try: order = int(line.split()[1])
                except: pass
            # Leer los pares pixel-longitud de onda
            if nlines and len(pixels) < nlines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        px = float(parts[0])
                        wl = float(parts[1])
                        if 3000 < wl < 11000:  # rango plausible en Angstroms
                            pixels.append(px)
                            wavelengths.append(wl)
                    except:
                        pass

        resultados.append({
            'nombre': nombre, 'rms': rms, 'nlines': nlines,
            'function': function, 'order': order,
            'pixels': pixels, 'wavelengths': wavelengths
        })

    # Plot resumen de RMS y nlines
    nombres = [r['nombre'][-20:] for r in resultados]
    rms_vals = [r['rms'] if r['rms'] else 0 for r in resultados]
    nlines_vals = [r['nlines'] if r['nlines'] else 0 for r in resultados]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(resultados)*0.8), 8))
    fig.suptitle('Calidad de la calibración en longitud de onda', fontsize=13)

    axes[0].bar(range(len(nombres)), rms_vals, color=['green' if r < 0.3 else 'red' for r in rms_vals])
    axes[0].axhline(0.3, color='r', linestyle='--', label='Límite recomendado 0.3 Ang')
    axes[0].set_xticks(range(len(nombres)))
    axes[0].set_xticklabels(nombres, rotation=45, ha='right', fontsize=7)
    axes[0].set_ylabel('RMS (Ang)')
    axes[0].set_title('RMS del ajuste por arco')
    axes[0].legend()

    axes[1].bar(range(len(nombres)), nlines_vals, color=['green' if n >= 20 else 'orange' for n in nlines_vals])
    axes[1].axhline(20, color='r', linestyle='--', label='Mínimo recomendado (20)')
    axes[1].set_xticks(range(len(nombres)))
    axes[1].set_xticklabels(nombres, rotation=45, ha='right', fontsize=7)
    axes[1].set_ylabel('N líneas identificadas')
    axes[1].set_title('Líneas identificadas por arco')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{work_dir}ARCS/diag_wl_calibration_quality.png', dpi=150)
    plt.close()

    # Plot solución pixel-lambda para el primer arco con datos
    for r in resultados:
        if len(r['pixels']) > 5:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            #fig.suptitle(f"Solución WL: {r['nombre']}  RMS={r['rms']:.3f} Ang  N={r['nlines']}", fontsize=11)
            axes[0].scatter(r['pixels'], r['wavelengths'], s=20, color='steelblue')
            axes[0].set_xlabel('Píxel')
            axes[0].set_ylabel('Longitud de onda (Ang)')
            axes[0].set_title('Solución pixel → λ')
            # Residuales aproximados
            if len(r['pixels']) > 3:
                coeffs = np.polyfit(r['pixels'], r['wavelengths'], 3)
                wl_fit = np.polyval(coeffs, r['pixels'])
                residuals = np.array(r['wavelengths']) - wl_fit
                axes[1].scatter(r['wavelengths'], residuals, s=20, color='steelblue')
                axes[1].axhline(0, color='r', linestyle='--')
                axes[1].set_xlabel('Longitud de onda (Ang)')
                axes[1].set_ylabel('Residual (Ang)')
                axes[1].set_title(f'Residuales  σ={np.std(residuals):.3f} Ang')
            plt.tight_layout()
            plt.savefig(f'{work_dir}ARCS/diag_wl_solucion_{r["nombre"]}.png', dpi=150)
            plt.close()
            break


#diag_wl_calibration(work_dir)



























notify('Comienza la seccion de FITCOORDS...')


if RS['17'][0] == True:
    notify('Ya se realizo la ejecucion de la tarea FITCOORDS')
else:
    original_dir = os.getcwd()
    arcs_dir = os.path.abspath(f'{work_dir}ARCS')
    os.chdir(arcs_dir)

    arc_files = [f for f in os.listdir(f'{work_dir}ARCS') if f.endswith(".fits") and 'cr' in f.lower()]
    arc_files.sort()

    for F in range(len(arc_files)):

        iraf.fitcoords(
            images = f"{arc_files[F].replace('.fits','')}",
            fitname = f"id{arc_files[F].replace('.fits','')}",
            interactive = "yes",
            combine = "no",
            database = "database",
            deletions = "deletions.db",
            function = "chebyshev",
            xorder = 4,
            yorder = 3,
            logfiles = "STDOUT,logfile",
            plotfile = "plotfile",
            graphics = "stdgraph",
        )

    os.chdir(original_dir)
    RS['17'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)




notify('Comienza la seccion de TRANSFORM para arcos...')


if RS['18'][0] == True:
    notify('Ya se realizo la ejecucion de la tarea TRANSFORM')
else:
    original_dir = os.getcwd()
    arcs_dir = os.path.abspath(f'{work_dir}ARCS')
    os.chdir(arcs_dir)

    arc_files = [f for f in os.listdir(f'{work_dir}ARCS') if f.endswith(".fits") and 'cr' in f.lower()]
    arc_files.sort()

    for F in range(len(arc_files)):

        iraf.transform( 
            input = f"{arc_files[F].replace('.fits','')}",
            output = f"{(arc_files[F].replace('.fits','')).replace('crLA','wlcal')}",
            fitnames = f"id{arc_files[F].replace('.fits','')}{arc_files[F].replace('.fits','')}",
            database = "database",
            interptype = "spline3"
        )

    os.chdir(original_dir)
    Plotganizer(work_dir).png_x_folder()
    shutil.copytree(str(f'{work_dir}ARCS/database/'), str(f'{work_dir}OBJS/database/'),dirs_exist_ok=True)
    RS['18'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)




notify('Se tiene que ingresar la tabla con el match entre Arcos y Objetos. Previamente se debe haber copiado la carpeta database a OBJS')




notify('Comienza la seccion de TRANSFORM para objetos...')


if RS['19'][0] == True:
    notify('Ya se realizo la ejecucion de la tarea TRANSFORM')
else:
    original_dir = os.getcwd()
    obj_dir = os.path.abspath(f'{work_dir}OBJS')
    os.chdir(obj_dir)


    obj_files = [f for f in os.listdir(f'{work_dir}OBJS') if f.endswith(".fits") and 'cr' in f.lower()]
    obj_files.sort()

    arc_files = [f for f in os.listdir(f'{work_dir}ARCS') if f.endswith(".fits") and 'cr' in f.lower()]
    arc_files.sort()

    #print(arc_files)

    ruta_association_file = os.path.join(f'{work_dir}OBJS/', 'association')

    shutil.move(str(f'{work_dir}association'), str(ruta_association_file))

    #while True:

        #if os.path.exists(ruta_association_file):
    with open(f'{work_dir}OBJS/association', 'r') as f:
        for linea in f:
            linea = linea.strip()
            
            if linea:
                dupla = ast.literal_eval(linea)

                obj_assos = [f for f in obj_files if dupla[0] in f][0]
                arc_assos = [f for f in arc_files if dupla[1] in f][0]

                print(obj_assos,arc_assos,'\n')

                iraf.transform( 
                    input = f"{obj_assos.replace('.fits','')}",
                    output = f"{(obj_assos.replace('.fits','')).replace('crLA','wlcal')}",
                    fitnames = f"id{arc_assos.replace('.fits','')}{arc_assos.replace('.fits','')}",
                    database = "database",
                    interptype = "spline3"
                )

                print('\n')

    os.chdir(original_dir)
    Plotganizer(work_dir).png_x_folder()
    RS['19'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)


        #print(f"\n[!] No encontré el archivo association en la carpeta.")
        #respuesta = input("¿Ya lo agregaste al folder? (Presiona ENTER para volver a buscar o escribe 'salir' para cancelar): ")
        
        # Opción por si decides abortar el script
        #if respuesta.lower().strip() == 'salir':
        #    print("Proceso cancelado por el usuario.")
        #    break
            
        #print("Buscando de nuevo...")
        #time.sleep(0.5) # Espera medio segundo antes de reintentar
        


#iraf.splot('/home/hollman/GEHR_specs/2023jun_spm/nite01/ARCS/Arc0008o_wlcal.fits')
#iraf.splot('/home/hollman/GEHR_specs/2023jun_spm/nite01/OBJS/N4214A0001o_wlcal.fits')



# ─────────────────────────────────────────────
# PASOS 18+19 — VERIFICACIÓN POST-TRANSFORM
# Qué buscar: líneas de arco perfectamente verticales,
#             líneas de cielo rectas en los objetos
# ─────────────────────────────────────────────
def diag_post_transform(work_dir):
    # Arcos transformados
    arcs_wlcal = sorted(glob.glob(f'{work_dir}ARCS/*wlcal*.fits'))
    objs_wlcal = sorted(glob.glob(f'{work_dir}OBJS/*wlcal*.fits'))

    def _plot_rectification(path, titulo, outpath, lineas_ref=None):
        with fits.open(path) as hdul:
            data = hdul[0].data
            header = hdul[0].header

        # Reconstruir eje de longitud de onda si existe WCS
        naxis1 = header.get('NAXIS1', data.shape[1])
        crval  = header.get('CRVAL1', 0)
        cdelt  = header.get('CDELT1', 1)
        crpix  = header.get('CRPIX1', 1)
        wl_axis = crval + (np.arange(naxis1) + 1 - crpix) * cdelt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(titulo, fontsize=12)

        vmin, vmax = np.percentile(data, [2, 98])
        axes[0].imshow(data, cmap='viridis', origin='lower', aspect='auto',
                       vmin=vmin, vmax=vmax,
                       extent=[wl_axis[0], wl_axis[-1], 0, data.shape[0]])
        axes[0].set_xlabel('Longitud de onda (Ang)')
        axes[0].set_ylabel('Fila Y (espacial)')
        axes[0].set_title('Imagen 2D rectificada\n[líneas deben ser VERTICALES]')
        if lineas_ref:
            for wl, nombre in lineas_ref:
                if wl_axis[0] < wl < wl_axis[-1]:
                    axes[0].axvline(wl, color='r', linestyle='--', alpha=0.6, lw=0.8)
                    axes[0].text(wl, data.shape[0]*0.02, nombre, color='r',
                                 fontsize=6, rotation=90, va='bottom')

        # Perfil espacial
        axes[1].plot(np.median(data, axis=1))
        axes[1].set_xlabel('Fila Y')
        axes[1].set_ylabel('Mediana (ADUs)')
        axes[1].set_title('Perfil espacial\n[fondo debe ser uniforme]')

        # Espectro colapsado (1D promedio)
        axes[2].plot(wl_axis, np.median(data, axis=0), lw=0.8, color='steelblue')
        axes[2].set_xlabel('Longitud de onda (Ang)')
        axes[2].set_ylabel('Flujo mediano')
        axes[2].set_title('Espectro 1D colapsado')
        if lineas_ref:
            for wl, nombre in lineas_ref:
                if wl_axis[0] < wl < wl_axis[-1]:
                    axes[2].axvline(wl, color='r', linestyle='--', alpha=0.7, lw=0.8)

        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()

    # Líneas de cielo y arco de referencia (ThAr comunes + cielo)
    lineas_sky = [
        (5577.34, '[OI]'), (5889.95, 'NaD1'), (5895.92, 'NaD2'),
        (6300.30, '[OI]'), (6363.78, '[OI]'), (6863.97, 'B-band'),
    ]
    lineas_emision = [
        (4861.33, 'Hβ'), (4958.91, '[OIII]'), (5006.84, '[OIII]'),
        (6548.05, '[NII]'), (6562.80, 'Hα'), (6583.45, '[NII]'),
        (6716.44, '[SII]'), (6730.82, '[SII]'),
    ]

    if arcs_wlcal:
        _plot_rectification(
            arcs_wlcal[0],
            f'Post-Transform ARCO: {Path(arcs_wlcal[0]).name}',
            f'{work_dir}ARCS/diag_transform_arc.png',
            lineas_ref=None
        )

    for obj_path in objs_wlcal[:2]:  # primeros 2 objetos
        nombre = Path(obj_path).stem
        _plot_rectification(
            obj_path,
            f'Post-Transform OBJETO: {nombre}',
            f'{work_dir}OBJS/diag_transform_{nombre}.png',
            lineas_ref=lineas_sky + lineas_emision
        )


#diag_post_transform(work_dir)

























notify('Comienza la busqueda de espectros calibrados y la conversion a 1D')


iraf.noao()
iraf.twodspec()
iraf.apextract()


if RS['20'][0] == True:
    notify('Ya se realizo la conversion a 1D de los objetos calibrados. Se pueden revisar y/o combinar')
else:

    obj_files = [f for f in os.listdir(f'{work_dir}OBJS') if f.endswith(".fits") and 'wlcal' in f.lower()]
    obj_files.sort()




    print(obj_files)

    for spec in obj_files:

        print(spec)

        iraf.apall(
            input =f"{work_dir}OBJS/{spec}"                 , #List of input images
            nfind =                 1, #Number of apertures to be found automatically
            output = f"{work_dir}OBJS/{spec.replace('wlcal','1d')}"             , #List of output spectra
            apertures = ""            , #Apertures
            format = "multispec"    , #Extracted spectra format
            references = ""             , #List of aperture reference images
            profiles = ""             , #List of aperture profile images\n
            interactive = "yes"            , #Run task interactively?
            find = "yes"            , #Find apertures?
            recenter = "yes"            , #Recenter apertures?
            resize = "yes"            , #Resize apertures?
            edit = "yes"            , #Edit apertures?
            trace = "yes"            , #Trace apertures?
            fittrace = "yes"            , #Fit the traced points interactively?
            extract = "yes"            , #Extract spectra?
            extras = "no"            , #Extract sky, sigma, etc.?
            review = "yes"            , #Review extractions?\n
            line = "INDEF"          , #Dispersion line
            nsum = 500, #500             , #Number of dispersion lines to sum or median\n\n# DEFAULT APERTURE PARAMETERS\n
            lower = -5.0            , #Lower aperture limit relative to center
            upper = 5.0             , #Upper aperture limit relative to center
            apidtable = ""             , #Aperture ID table (optional)\n\n# DEFAULT BACKGROUND PARAMETERS\n
            b_function = "chebyshev"    , #Background function
            b_order = 4              , #Background function order
            b_sample = "-180:-100,80:170"  , #Background sample regions
            b_naverage = -4             , #Background average or median
            b_niterate = 1              , #Background rejection iterations
            b_low_reject = 3.0             , #Background lower rejection sigma
            b_high_rejec = 3.0             , #Background upper rejection sigma
            b_grow = 0.0             , #Background rejection growing radius\n\n# APERTURE CENTERING PARAMETERS\n
            width = 5.0             , #Profile centering width
            radius = 10.0            , #Profile centering radius
            threshold = 10.0             , #Detection threshold for profile centering\n\n# AUTOMATIC FINDING AND ORDERING PARAMETERS\n
            minsep = 3.0             , #Minimum separation between spectra
            maxsep = 100000.0        , #Maximum separation between spectra
            order = "increasing"   , #Order of apertures\n\n# RECENTERING PARAMETERS\n
            aprecenter = ""             , #Apertures for recentering calculation
            npeaks = "INDEF"          , #Select brightest peaks
            shift = "no"            , #Use average shift instead of recentering?\n\n# RESIZING PARAMETERS\n
            llimit = "INDEF"          , #Lower aperture limit relative to center
            ulimit = "INDEF"          , #Upper aperture limit relative to center
            ylevel = 0.1            , #Fraction of peak or intensity for automatic width
            peak = "yes"            , #Is ylevel a fraction of the peak?
            bkg = "yes"            , #Subtract background in automatic width?
            r_grow = 0.0             , #Grow limits by this factor
            avglimits = "no"             , #Average limits over all apertures?\n\n# TRACING PARAMETERS\n
            t_nsum = 20             , #Number of dispersion lines to sum
            t_step = 10             , #Tracing step
            t_nlost = 3              , #Number of consecutive times profile is lost before quitting
            t_function = "legendre"     , #Trace fitting function
            t_order = 3              , #Trace fitting function order
            t_sample = "*"            , #Trace sample regions
            t_naverage = 1              , #Trace average or median
            t_niterate = 1              , #Trace rejection iterations
            t_low_reject = 2.5             , #Trace lower rejection sigma
            t_high_rejec = 2.5             , #Trace upper rejection sigma
            t_grow = 0.0             , #Trace rejection growing radius\n\n# EXTRACTION PARAMETERS\n
            background = "median"         , #Background to subtract
            skybox = 1              , #Box car smoothing length for sky
            weights = "none"         , #Extraction weights (none|variance)
            pfit = "fit1d"        , #Profile fitting type (fit1d|fit2d)
            clean = "no"             , #Detect and replace bad pixels?
            saturation = "INDEF"          , #Saturation level
            readnoise = "RDNOISE"           , #Read out noise sigma (photons)
            gain = "GAIN"           , #Photon gain (photons/data number)
            lsigma = 4.             , #Lower rejection threshold
            usigma = 4.             , #Upper rejection threshold
            nsubaps = 1              , #Number of subapertures per aperture
        )            
    RS['20'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)
    






# ─────────────────────────────────────────────
# PASO 20 — ESPECTROS 1D EXTRAÍDOS (mejorado)
# Qué buscar: líneas de emisión en las posiciones correctas,
#             continuo suave, S/N razonable
# ─────────────────────────────────────────────
def diag_1d_spectra(work_dir):
    spec_files = sorted(glob.glob(f'{work_dir}OBJS/*1d*.fits'))

    lineas = {
        'Hβ':     4861.33,
        '[OIII]': 5006.84,
        'Hα':     6562.80,
        '[NII]':  6583.45,
        '[SII]a': 6716.44,
        '[SII]b': 6730.82,
    }

    for path in spec_files:
        with fits.open(path) as hdul:
            data   = hdul[0].data
            header = hdul[0].header

        flujo = data[0] if data.ndim > 1 else data
        naxis1 = header.get('NAXIS1', len(flujo))
        crval  = header.get('CRVAL1', 0)
        cdelt  = header.get('CDELT1', 1)
        crpix  = header.get('CRPIX1', 1)
        wl = crval + (np.arange(naxis1) + 1 - crpix) * cdelt

        # Estimación simple de S/N (señal / ruido del continuo)
        continuo_mask = (wl > 5200) & (wl < 5500)
        if continuo_mask.sum() > 10:
            snr = np.median(flujo[continuo_mask]) / np.std(flujo[continuo_mask])
        else:
            snr = np.nan

        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        nombre = header.get('OBJECT', Path(path).stem)
        fig.suptitle(f'{nombre}  —  S/N continuo ≈ {snr:.1f}', fontsize=13)

        # Espectro completo
        axes[0].step(wl, flujo, where='mid', color='black', lw=0.8)
        for nombre_l, wl_l in lineas.items():
            if wl[0] < wl_l < wl[-1]:
                axes[0].axvline(wl_l, color='r', linestyle='--', alpha=0.6, lw=0.8)
                axes[0].text(wl_l, axes[0].get_ylim()[1]*0.9, nombre_l,
                             color='r', fontsize=7, rotation=90, va='top')
        axes[0].set_xlabel('Longitud de onda (Ang)')
        axes[0].set_ylabel('Flujo (ADUs)')
        axes[0].set_title('Espectro completo')
        axes[0].grid(alpha=0.3)

        # Zoom en Hα+[NII]
        mask_ha = (wl > 6480) & (wl < 6750)
        if mask_ha.sum() > 5:
            axes[1].step(wl[mask_ha], flujo[mask_ha], where='mid', color='steelblue', lw=1)
            for nombre_l, wl_l in lineas.items():
                if 6480 < wl_l < 6750:
                    axes[1].axvline(wl_l, color='r', linestyle='--', alpha=0.7)
                    axes[1].text(wl_l, axes[1].get_ylim()[1]*0.9, nombre_l,
                                 color='r', fontsize=8, rotation=90, va='top')
            axes[1].set_xlabel('Longitud de onda (Ang)')
            axes[1].set_ylabel('Flujo (ADUs)')
            axes[1].set_title('Zoom Hα + [NII] + [SII]')
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        outname = path.replace('.fits', '_diag.png')
        plt.savefig(outname, dpi=150)
        plt.close()
        print(f'  → {Path(path).name}  S/N≈{snr:.1f}')


#diag_1d_spectra(work_dir)
































notify('Ploteando espectros 1D extraidos')


if RS['21'][0] == True:
    notify('Los espectros 1D extraidos ya estan ploteados')
else:
    def plot_espectro_1d(ruta_fits):
        with fits.open(ruta_fits) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            # 1. Obtener el flujo (eje Y)
            # apall a veces guarda varias dimensiones (flujo, background, etc.)
            # Si es extracción estándar, el flujo está en la primera dimensión
            if data.ndim > 1:
                flujo = data[0] 
            else:
                flujo = data

            # 2. Reconstruir la longitud de onda (eje X) usando el Header
            # IRAF usa: Wavelength = CRVAL1 + (pixel - CRPIX1) * CDELT1
            try:
                n_pixeles = header['NAXIS1']
                crval = header['CRVAL1']
                cdelt = header['CDELT1']
                crpix = header.get('CRPIX1', 1) # Por defecto 1 si no existe
                
                # Crear el arreglo de longitud de onda
                referencia_pix = np.arange(n_pixeles) + 1
                longitud_onda = crval + (referencia_pix - crpix) * cdelt
                
            except KeyError:
                print("No se encontraron las llaves WCS. Ploteando en píxeles.")
                longitud_onda = np.arange(len(flujo))

            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot()
            ax.step(longitud_onda, flujo, where='mid', color='black', lw=1)
            ax.axvline(x=6564.6, color='r', linestyle='--')
            ax.set_xlabel(f"Longitud de Onda ({header.get('CUNIT1', 'Ang')})")
            ax.set_ylabel("Flujo (ADUs / Electrones)")
            ax.set_title(f"Espectro: {header.get('OBJECT', 'Sin nombre')}")
            ax.grid(alpha=0.3)
            fig.savefig(ruta_fits.replace('.fits', '.png'),dpi=150)
            plt.close(fig)

    _1D = [f for f in os.listdir(f'{work_dir}OBJS') if f.endswith(".fits") and '1d' in f.lower()]
    _1D.sort()

    for spec in _1D:
        plot_espectro_1d(f'{work_dir}OBJS/{spec}')

    RS['21'][0] = True
    guardar_logfile(f'{work_dir}/log_reduc',RS)    














guardar_logfile(f'{work_dir}/log_reduc',RS)

