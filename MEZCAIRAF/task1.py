from pyraf import iraf
from pyraf import gki
from utils import leer_o_crear_logfile
from utils import notify
from utils import guardar_logfile
from utils import list_generator
from utils import Plotganizer
from utils import task_checker

def MASTERBIAS(work_dir):


    notify('BIAS',mode='M')
    RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')

    iraf.noao()
    iraf.imred()
    iraf.ccdred()

    notify('Comienza creacion del MASTERFLAT')
    notify('Generacion de listas de bias')



    task_verifier = task_checker(1,RS)
    if RS['1'][0] == False:
        try:
            list_generator(FOLDER=f'{work_dir}BIAS', 
                        List_in='Bias_raw', 
                        List_out='Bias_trim', 
                        out_key='trim',
                        img_type='bias')
            RS['1'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['1'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['1'][1]}')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['1'][1]} porque hay tareas anteriores pendientes.')









    task_verifier = task_checker(2,RS)
    if RS['2'][0] == True:
        notify('Los bias ya estan cortados con CCDPROC')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['2'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:
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
        except:
            RS['2'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['2'][1]}')




    task_verifier = task_checker(3,RS)
    if RS['3'][0] == True:
        notify('El masterbias existe')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['3'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:
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
        except:
            RS['3'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['3'][1]}')



