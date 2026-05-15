from pyraf import iraf
from pyraf import gki
from utils import leer_o_crear_logfile
from utils import notify
from utils import guardar_logfile


def SET_INSTRUMENT(work_dir):

    notify('SET INST',mode='M')

    RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')
    notify('Cargando librerias esenciales')
    iraf.noao()
    iraf.imred()
    iraf.ccdred()
    iraf.set(stdgraph="xgterm")
    notify('Fijar el instrumento para CCDPROC')



    if RS['0'][0] == True:
        notify('Ya esta cargado el instrumento')
    else:
        try:
            notify('Fijar el instrumento y salir')
            iraf.setinstrument(instrument = 'mezcal' ,site = 'ospm' ,directory = 'ccddb$')
            RS['0'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['0'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
