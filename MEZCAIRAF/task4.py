from pyraf import iraf
from pyraf import gki
from utils import leer_o_crear_logfile
from utils import notify
from utils import guardar_logfile
from utils import list_generator
from utils import Plotganizer
from utils import task_checker
from joblib import Parallel, delayed
from cosmicray_cleaner import cosmicos_cleaner
import os


def SPECS(work_dir):

    notify('SPECS',mode='M')
    RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')

    iraf.noao()
    iraf.imred()
    iraf.ccdred()

    notify('Comienza el corte de los Objetos')
    notify('Generacion de listas de Objetos')


    task_verifier = task_checker(12,RS)
    if RS['12'][0] == False:
        try:
            list_generator(FOLDER=f'{work_dir}OBJS', 
                        List_in='Obj_raw', 
                        List_out='Obj_trim', 
                        out_key='trim',
                        img_type='o')
            RS['12'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['12'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['12'][1]}')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['12'][1]} porque hay tareas anteriores pendientes.')





    task_verifier = task_checker(13,RS)
    if RS['13'][0] == True:
        notify('Los objetos ya estan cortados con CCDPROC + masterbias + FlatNmaster.fits')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['13'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:
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
        except:
            RS['13'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['13'][1]}')



    notify('Generacion de listas de objetos (sin cosmicos)')

    task_verifier = task_checker(14,RS)
    if RS['14'][0] == False:
        try:
            list_generator(FOLDER=f'{work_dir}OBJS', 
                        List_in='Obj_raw', 
                        List_out='Obj_cr', 
                        out_key='crLa',
                        img_type='trim')
            RS['14'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['14'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)            
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['14'][1]}')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['14'][1]} porque hay tareas anteriores pendientes.')


    notify('Comienza la remocion de cosmicos de los Objetos ')

    task_verifier = task_checker(15,RS)
    if RS['15'][0] == True:
        notify('Ya se removieron los cosmicos de los arcos')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['15'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:
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
        except:
            RS['15'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['15'][1]}')