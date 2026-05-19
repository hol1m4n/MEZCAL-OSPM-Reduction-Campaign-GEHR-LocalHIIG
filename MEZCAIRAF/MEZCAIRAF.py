import os
from pathlib import Path
import shutil



from utils import notify
from utils import Plotganizer
from utils import leer_o_crear_logfile

from task0 import SET_INSTRUMENT
from task1 import MASTERBIAS
from task2 import MASTERFLAT
from task3 import ARCOS
from task4 import SPECS
from task5 import WAVELENGHT_CALIBRATION
from task6 import ONE_DIMENTIONAL_SPECTRUM_EXTRACT

from summary_plots import summary_plotter 


notify('START',mode='M')

work_dir = '/home/hollman/GEHR_specs/2023jun_spm/nite06/'


notify(f'Trabajando en {work_dir}')
notify('Organizacion de archivos y asignacion de headers')
Plotganizer(work_dir).executor()
notify('Conversion de imagenes crudas .fits a .png')
Plotganizer(work_dir).png_x_folder()

RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')


SET_INSTRUMENT(work_dir)
MASTERBIAS(work_dir)
MASTERFLAT(work_dir)
ARCOS(work_dir)
SPECS(work_dir)
WAVELENGHT_CALIBRATION(work_dir)
ONE_DIMENTIONAL_SPECTRUM_EXTRACT(work_dir)

notify('END',mode='M')




RS = leer_o_crear_logfile(f'{work_dir}/log_reduc') # Se lee nuevamente el log_reduc y si todo termino con exito entonces ahora si se plotea el summary
plotear_resumen = all(valor[0] is True for valor in RS.values())
if plotear_resumen:
    summary_plotter(work_dir)

    
















