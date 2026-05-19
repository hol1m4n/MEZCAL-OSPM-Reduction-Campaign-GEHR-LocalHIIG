from pyraf import iraf
from pyraf import gki
from utils import leer_o_crear_logfile
from utils import notify
from utils import guardar_logfile
from utils import list_generator
from utils import Plotganizer
from utils import task_checker
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


    notify('Comienza el identify basado en el primer arco. Se hace el identify para cada uno de ellos y se propaga a los subsiguientes, pertenecientes a cada objeto.')

    task_verifier = task_checker(16,RS)
    if RS['16'][0] == True:
        notify('Ya se realizo la calibracion en longitud de onda, revisar el database')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['16'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:

            notify('RECORDAR USAR m l d f PARA AJUSTAR DE MANERA APROPIADA!!!!')

            notify('RECORDAR BORRAR PUNTOS CON RESIDUALES MAYORES A 0.1 ANG!!!!')

            original_dir = os.getcwd()
            arcs_dir = os.path.abspath(f'{work_dir}ARCS')
            os.chdir(arcs_dir)


            arc_files = [f for f in os.listdir(f'{work_dir}ARCS') if f.endswith(".fits") and 'cr' in f.lower()]
            arc_files.sort()

            '''
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
            

            mv_cal_groups = [f for f in os.listdir(f'{work_dir}') if 'waveleng_calg' in f.lower()]
            mv_cal_groups.sort()

            WL_cal_groups = [f for f in os.listdir(f'{work_dir}ARCS') if 'waveleng_calg' in f.lower()]

            if len(mv_cal_groups) == 0 and len(WL_cal_groups) == 0:
                #notify('No hay grupos de arcos para la calibracion de longitud de onda, crearlos o moverlos.')
            #elif len(mv_cal_groups) > 0 and len(WL_cal_groups) == 0:
                for e in mv_cal_groups:
                    ruta_wlcal_file = os.path.join(f'{work_dir}ARCS/', f'{e}')
                    shutil.move(str(f'{work_dir}{e}'), str(ruta_wlcal_file))

            '''

            wlcal_start = input('Ya se movieron los archivos waveleng_calg a ARCS?(y/n)')
            if wlcal_start == 'y':
                print('Continuamos...')         

            WL_cal_groups = [f for f in os.listdir(f'{work_dir}ARCS') if 'waveleng_calg' in f.lower()]
            WL_cal_groups.sort()

            for G in range(len(WL_cal_groups)):
                grupo_n = []
                with open(f'{work_dir}ARCS/WaveLeng_calG{G+1}', "r") as arc_individual:
                    for line in arc_individual:
                        #print(line)
                        grupo_n.append(line.strip())
                    #print("\n")
                chosen = grupo_n[0] # For identify, the rest are reidentify 
                grupo_n.remove(chosen) # Aqui se remueve el primero que es con el que se calibra
                print(grupo_n, chosen,"\n")

                iraf.identify(
                    images = f'{chosen}', #{work_dir}ARCS/
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

                if len(grupo_n) > 1:
                    with open(f'{work_dir}ARCS/reidentify_{G+1}', "w") as reidentify_list:
                        for i in grupo_n:
                            reidentify_list.write(f'{i} \n')

                    iraf.reidentify(reference = f'{chosen}', # {work_dir}ARCS/
                        images = f'@reidentify_{G+1}', #{work_dir}ARCS/
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
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['16'][1]}')


    notify('Comienza la seccion de FITCOORDS...')

    task_verifier = task_checker(17,RS)
    if RS['17'][0] == True:
        notify('Ya se realizo la ejecucion de la tarea FITCOORDS')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['17'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:
            notify('RECORDAR BORRAR PUNTOS MALOS DEL AJUSTE CON d p Y LUEGO f !!!!')


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
                notify(f'X X X',mode='M')
                notify(f'Hubo un fallo en la tarea {RS['17'][1]}')


        except:
            RS['17'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)            
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['17'][1]}')


    notify('Comienza la seccion de TRANSFORM para arcos...')

    task_verifier = task_checker(18,RS)
    if RS['18'][0] == True:
        notify('Ya se realizo la ejecucion de la tarea TRANSFORM')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['18'][1]} porque hay tareas anteriores pendientes.')
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
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['18'][1]}')


    notify('Comienza la seccion de TRANSFORM para objetos...')

    


    try:
        if not os.path.exists(os.path.join(f'{work_dir}OBJS/', 'association')):
            ruta_association_file = os.path.join(f'{work_dir}OBJS/', 'association')
            shutil.move(str(f'{work_dir}association'), str(ruta_association_file))
    except:
        notify('El archivo association no se encuentra ni el directorio de trabajo ni en la carpeta OBJS. Crear el objeto o moverlo al directorio.Se tiene que ingresar la tabla con el match entre Arcos y Objetos. Previamente se debe haber copiado la carpeta database a OBJS')


    task_verifier = task_checker(19,RS)
    if RS['19'][0] == True:
        notify('Ya se realizo la ejecucion de la tarea TRANSFORM')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['19'][1]} porque hay tareas anteriores pendientes.')
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
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['19'][1]}')










