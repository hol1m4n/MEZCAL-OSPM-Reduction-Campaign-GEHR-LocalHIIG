from pyraf import iraf
from pyraf import gki
from utils import leer_o_crear_logfile
from utils import notify
from utils import guardar_logfile
from utils import list_generator
from utils import Plotganizer
from joblib import Parallel, delayed
from cosmicray_cleaner import cosmicos_cleaner
import os

def ARCOS(work_dir):


    notify('ARCS',mode='M')
    RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')

    iraf.noao()
    iraf.imred()
    iraf.ccdred()

    notify('Comienza el corte de los Arcos ')
    notify('Generacion de listas de arcos')

    if RS['8'][0] == False:
        try:
            list_generator(FOLDER=f'{work_dir}ARCS', 
                        List_in='Arcs_raw', 
                        List_out='Arcs_trim', 
                        out_key='trim',
                        img_type='arc')
            RS['8'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['8'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)



    if RS['9'][0] == True:
        notify('Los arcos ya estan cortados con CCDPROC + masterbias + FlatNmaster.fits')
    else:
        try:
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
        except:
            RS['9'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)



    
    notify('Generacion de listas de arcos (sin cosmicos)')


    if RS['10'][0] == False:
        try:
            list_generator(FOLDER=f'{work_dir}ARCS', 
                        List_in='Arcs_raw', 
                        List_out='Arcs_cr', 
                        out_key='crLa',
                        img_type='trim')
            RS['10'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['10'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)



    notify('Comienza la remocion de cosmicos de los Arcos ')



    if RS['11'][0] == True:
        notify('Ya se removieron los cosmicos de los arcos')
    else:
        try:
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
        except:
            RS['11'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)            






