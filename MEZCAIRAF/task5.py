from pyraf import iraf
from pyraf import gki
from utils import leer_o_crear_logfile
from utils import notify
from utils import guardar_logfile
from utils import list_generator
from utils import Plotganizer
import os
import random
import shutil
import ast

def WAVELENGHT_CALIBRATION(work_dir):

    notify('WL CAL',mode='M')
    RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')

    iraf.noao()
    iraf.twodspec()
    iraf.longslit()


    notify('Comienza el identify basado en 4 arcos aleatorios. Se hace el identify para cada uno de ellos y se propaga a los subsiguientes.')


    if RS['16'][0] == True:
        notify('Ya se realizo la calibracion en longitud de onda, revisar el database')
    else:
        try:
            original_dir = os.getcwd()
            arcs_dir = os.path.abspath(f'{work_dir}ARCS')
            os.chdir(arcs_dir)


            arc_files = [f for f in os.listdir(f'{work_dir}ARCS') if f.endswith(".fits") and 'cr' in f.lower()]
            arc_files.sort()

            integ_ = 4
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
        except:
            RS['16'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)



    notify('Comienza la seccion de FITCOORDS...')


    if RS['17'][0] == True:
        notify('Ya se realizo la ejecucion de la tarea FITCOORDS')
    else:
        try:
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

            fitcoords_status = input('Fitcoords salio bien?(y/n)')

            if fitcoords_status == 'y':
                RS['17'][0] = True
                guardar_logfile(f'{work_dir}/log_reduc',RS)
            if fitcoords_status == 'n':
                RS['17'][0] = False
                guardar_logfile(f'{work_dir}/log_reduc',RS)


        except:
            RS['17'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)            



    notify('Comienza la seccion de TRANSFORM para arcos...')


    if RS['18'][0] == True:
        notify('Ya se realizo la ejecucion de la tarea TRANSFORM')
    else:
        try:
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
        except:
            RS['18'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)            



    notify('Comienza la seccion de TRANSFORM para objetos...')

    


    try:
        if not os.path.exists(os.path.join(f'{work_dir}OBJS/', 'association')):
            ruta_association_file = os.path.join(f'{work_dir}OBJS/', 'association')
            shutil.move(str(f'{work_dir}association'), str(ruta_association_file))
    except:
        notify('El archivo association no se encuentra ni el directorio de trabajo ni en la carpeta OBJS. Crear el objeto o moverlo al directorio.Se tiene que ingresar la tabla con el match entre Arcos y Objetos. Previamente se debe haber copiado la carpeta database a OBJS')



    if RS['19'][0] == True:
        notify('Ya se realizo la ejecucion de la tarea TRANSFORM')
    else:
        try:
            original_dir = os.getcwd()
            obj_dir = os.path.abspath(f'{work_dir}OBJS')
            os.chdir(obj_dir)

            obj_files = [f for f in os.listdir(f'{work_dir}OBJS') if f.endswith(".fits") and 'cr' in f.lower()]
            obj_files.sort()

            arc_files = [f for f in os.listdir(f'{work_dir}ARCS') if f.endswith(".fits") and 'cr' in f.lower()]
            arc_files.sort()


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
        except:
            RS['19'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)











