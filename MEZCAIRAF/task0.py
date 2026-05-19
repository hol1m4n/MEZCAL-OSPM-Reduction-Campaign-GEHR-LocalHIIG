from pyraf import iraf
from pyraf import gki
from utils import leer_o_crear_logfile
from utils import notify
from utils import guardar_logfile
from utils import task_checker



def SET_INSTRUMENT(work_dir):

    notify('SET INST',mode='M')

    RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')
    notify('Cargando librerias esenciales')
    iraf.noao()
    iraf.imred()
    iraf.ccdred()
    iraf.set(stdgraph="xgterm")
    notify('Fijar el instrumento para CCDPROC')


    task_verifier = task_checker(0,RS)
    if RS['0'][0] == True:
        notify('Ya esta cargado el instrumento')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['0'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:
            notify('Fijar el instrumento y salir')
            iraf.setinstrument(instrument = 'mezcal' ,site = 'ospm' ,directory = 'ccddb$')
            RS['0'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['0'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['0'][1]}')
