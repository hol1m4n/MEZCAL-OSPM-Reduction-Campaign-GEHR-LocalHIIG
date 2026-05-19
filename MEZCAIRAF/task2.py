from pyraf import iraf
from pyraf import gki
from utils import leer_o_crear_logfile
from utils import notify
from utils import guardar_logfile
from utils import list_generator
from utils import Plotganizer
from utils import task_checker
from scipy.ndimage import median_filter
import numpy as np
from astropy.io import fits


def MASTERFLAT(work_dir):


    notify('FLATS',mode='M')
    RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')

    iraf.noao()
    iraf.imred()
    iraf.ccdred()

    notify('Generacion de listas de flats')

    task_verifier = task_checker(4,RS)
    if RS['4'][0] == False:
        try:
            list_generator(FOLDER=f'{work_dir}FLATS', 
                        List_in='Flat_raw', 
                        List_out='Flat_trim', 
                        out_key='trim',
                        img_type='flat')
            RS['4'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['4'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['4'][1]}')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['4'][1]} porque hay tareas anteriores pendientes.')







    task_verifier = task_checker(5,RS)
    if RS['5'][0] == True:
        notify('Los flats ya estan cortados con CCDPROC + masterbias')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['5'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:
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
        except:
            RS['5'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['5'][1]}')











    task_verifier = task_checker(6,RS)
    if RS['6'][0] == True:
        notify('El masterflat existe')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['6'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:
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
        except:
            RS['6'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)            
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['6'][1]}')



    iraf.noao()
    iraf.twodspec()
    iraf.longslit()
    iraf.set(stdgraph="xgterm")






    task_verifier = task_checker(7,RS)
    if RS['7'][0] == True:
        notify('El masterflat normalizado existe')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['7'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:
            notify('Enmascarando la fuga de luz... ')
            with fits.open(f'{work_dir}FLATS/masterflat.fits') as hdul:
                data = hdul[0].data.copy()
                header = hdul[0].header

            fuga_mask = np.zeros(data.shape, dtype=bool)
            fuga_mask[:,275:332] = True 
            data_smooth = median_filter(data, size=(1, 200)) 
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
            print(f'Mediana global del flat: {global_median:.4f}')

            fits.writeto(f'{work_dir}FLATS/FlatNmaster.fits', 
                        data / global_median, header, overwrite=True)

            Plotganizer(work_dir).png_x_folder()
            RS['7'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['7'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['7'][1]}')













































































