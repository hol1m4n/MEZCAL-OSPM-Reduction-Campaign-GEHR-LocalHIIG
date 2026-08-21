import os
import astropy.units as u
from astropy.coordinates import FK5, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

import io
import requests
import astropy.units as u
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


from joblib import Parallel, delayed
from utils import leer_o_crear_logfile
from utils import notify
from utils import guardar_logfile
from utils import task_checker

def sky_location_helper(file):

    try:
        astrometry_api_key = "wqkewwobfpwlvetj"
        astrometry = AstrometryNet()
        astrometry.api_key = astrometry_api_key

        archivo_fits = file
        directorio, nombre_base = os.path.split(archivo_fits)
        archivo_corregido = os.path.join(directorio, "Mirror_" + nombre_base)

        print(file,archivo_corregido,'\n')

        if not os.path.exists(archivo_corregido) and not 'mirror'in archivo_fits.lower() and not os.path.exists(archivo_corregido.replace(".fits", ".png")):
            with fits.open(archivo_fits) as hdul:
                header = hdul[0].header
                datos_imagen = hdul[0].data

            ra_sex = header["RA"]
            dec_sex = header["DEC"]
            epoch_val = header.get("EPOCH")

            c_original = SkyCoord(ra=ra_sex, dec=dec_sex, unit=(u.hourangle, u.deg), frame=FK5(equinox=f"J{epoch_val}"))
            ra_estimada = c_original.ra.degree
            dec_estimada = c_original.dec.degree

            print("--- COORDENADAS TEÓRICAS (DEL FITS) ---")
            print(f"Buscando cerca de RA: {ra_estimada:.5f}° | DEC: {dec_estimada:.5f}°")

            configuracion = {
                "center_ra": ra_estimada,
                "center_dec": dec_estimada,
                "radius": 0.5,
                "scale_units": "arcsecperpix",
                "scale_lower": 0.1,  
                "scale_upper": 0.5,  
            }

            try:
                wcs_header = astrometry.solve_from_image(archivo_fits, **configuracion)
                
                if wcs_header:
                    print("\n¡ÉXITO! Astrometría resuelta.")
                    
                    wcs_corregido = WCS(wcs_header)
                    alto_px, ancho_px = datos_imagen.shape
                    ra_real, dec_real = wcs_corregido.pixel_to_world_values(ancho_px / 2.0, alto_px / 2.0)
                    
                    print("\n--- NUEVAS COORDENADAS REALES (CORREGIDAS) ---")
                    print(f"Centro Real -> RA: {ra_real:.5f}° | DEC: {dec_real:.5f}°")
                    print(f"Desfase real corregido -> Delta RA: {ra_real - ra_estimada:.5f}° | Delta DEC: {dec_real - dec_estimada:.5f}°")
                    
                    nuevo_header = header.copy()
                    
                    nuevo_header.update(wcs_corregido.to_header())
                    
                    fits.writeto(archivo_corregido, datos_imagen, nuevo_header, overwrite=True)
                    print(f"\nArchivo guardado exitosamente en: {archivo_corregido}")
                    
                else:
                    print("\nNo se pudo resolver la astrometría. Verifica los márgenes de escala o el contraste.")
                    print(f"\nOcurrió un error en la astrometría del objeto. No se guardo la imagen: {e} \n")
                    archivo_fits = file
                    directorio, nombre_base = os.path.split(archivo_fits)
                    archivo_corregido = os.path.join(directorio, "Mirror_" + nombre_base)

                    fig, ax = plt.subplots(figsize=(50, 30))

                    ax.axis('off')

                    # Añade el texto en la coordenada (x=0.5, y=0.5) centrado
                    ax.text(0.5, 0.5, f"\nOcurrió un error en la astrometría del objeto. No se guardo la imagen: {e}", fontsize=24, color='black',
                            ha='center', va='center')

                    plt.savefig(f'{archivo_corregido.replace(".fits", ".png")}')

            except Exception as e:
                print(f"\nOcurrió un error en la astrometría del objeto. No se guardo la imagen: {e} \n")
                archivo_fits = file
                directorio, nombre_base = os.path.split(archivo_fits)
                archivo_corregido = os.path.join(directorio, "Mirror_" + nombre_base)

                fig, ax = plt.subplots(figsize=(50, 30))

                ax.axis('off')

                # Añade el texto en la coordenada (x=0.5, y=0.5) centrado
                ax.text(0.5, 0.5, f"\nOcurrió un error en la astrometría del objeto. No se guardo la imagen: {e}", fontsize=24, color='black',
                        ha='center', va='center')

                plt.savefig(f'{archivo_corregido.replace(".fits", ".png")}')







            fig = plt.figure(figsize=(15, 8))
            # --- Imagen FITS corregida ---

            archivo_fits_corregido = archivo_corregido

            with fits.open(archivo_fits_corregido) as hdul:
                header = hdul[0].header
                datos_imagen = hdul[0].data
                wcs_real = WCS(header)

            alto_px, ancho_px = datos_imagen.shape
            centro_x_px = ancho_px / 2.0
            centro_y_px = alto_px / 2.0

            ra_real, dec_real = wcs_real.pixel_to_world_values(centro_x_px, centro_y_px)

            print("--- INFORMACIÓN DEL ARCHIVO CALIBRADO ---")
            print(f"Objeto: {header.get('OBJECT', 'Desconocido')}")
            print(f"Centro Real de la imagen -> RA: {ra_real:.5f}° | DEC: {dec_real:.5f}°")

            norma = simple_norm(datos_imagen, stretch="asinh", percent=99.7)


            ax1 = fig.add_subplot(1, 2, 1, projection=wcs_real)
            im1 = ax1.imshow(datos_imagen, cmap="inferno", origin="lower", norm=norma)

            ax1.set_xlim(-0.5, ancho_px - 0.5)
            ax1.set_ylim(-0.5, alto_px - 0.5)

            ax1.plot(
                centro_x_px,
                centro_y_px,
                color="lime",
                marker="+",
                markersize=20,
                markeredgewidth=2,
                label="Centro Real",
            )

            ax1.set_xlabel("Ascensión Recta (RA)")
            ax1.set_ylabel("Declinación (DEC)")

            #plt.colorbar(im1, ax=ax1)
            plt.grid(color="white", ls="dashed", alpha=0.3)

            escala_x, escala_y = proj_plane_pixel_scales(wcs_real)
            alto_px, ancho_px = datos_imagen.shape
            fov_x_grados = ancho_px * escala_x
            fov_y_grados = alto_px * escala_y
            fov_x_minutos = fov_x_grados * 60
            fov_y_minutos = fov_y_grados * 60

            print("--- DIMENSIONES PRECISAS DEL FOV ---")
            print(f"FOV en X: {fov_x_grados:.4f}° ({fov_x_minutos:.2f} minutos de arco)")
            print(f"FOV en Y: {fov_y_grados:.4f}° ({fov_y_minutos:.2f} minutos de arco)")
            print(f"Tamaño total del campo: {fov_x_minutos:.2f}' x {fov_y_minutos:.2f}'")


            # --- Imagen a color de SDSS ---


            ax2 = fig.add_subplot(1, 2, 2)

            ra_real = ra_real * 1           
            dec_real = dec_real * 1         
            fov_x_minutos = fov_x_minutos * 1           
            fov_en_grados = (fov_x_minutos * u.arcmin).to(u.deg).value

            url = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"

            encuestas = [
                {"nombre": "SDSS", "id": "CDS/P/SDSS9/color-alt"},
                {"nombre": "DSS2", "id": "CDS/P/DSS2/color"},
                {"nombre": "DESI Legacy Survey", "id": "CDS/P/DESI-Legacy-Surveys/DR10/color"}
            ]

            img_color = None
            encuesta_usada = "Ninguna"

            for encuesta in encuestas:
                print(f"Intentando descargar imagen desde {encuesta['nombre']}...")
                params = {
                    "hips": encuesta["id"],
                    "ra": ra_real,
                    "dec": dec_real,
                    "fov": fov_en_grados,
                    "width": ancho_px,
                    "height": alto_px,
                    "projection": "TAN",
                    "coordsys": "icrs",
                    "format": "png"
                }

                try:
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        # Validar si la respuesta es una imagen real y no un texto de error del servidor
                        if "image" in response.headers.get("Content-Type", ""):
                            img_color = Image.open(io.BytesIO(response.content))
                            encuesta_usada = encuesta["nombre"]
                            print(f"¡Éxito! Imagen obtenida de {encuesta_usada}.\n")
                            #break
                        else:
                            print(f"Área fuera de cobertura en {encuesta['nombre']}. Probando siguiente opción...")
                    else:
                        print(f"Error de red en {encuesta['nombre']}. Status: {response.status_code}")
                except Exception as e:
                    print(f"Error al procesar {encuesta['nombre']}: {e}")

                # Dibujar la imagen encontrada en el subplot
                if img_color is not None:
                    im2 = ax2.imshow(img_color)
                    ax2.plot(
                        centro_x_px, centro_y_px, 
                        color="lime", marker="+", markersize=20, markeredgewidth=2
                    )
                    #ax2.set_title(f"Referencia: {encuesta_usada}", fontsize=14)

                    print(f'======> COORDENADAS REALES  RA:{ra_real} , DEC: {dec_real}\n')
                    print(f'{ra_real} {dec_real}\n')
                    ax2.text(
                        0.98,
                        0.05,
                        f'RA:{ra_real} , DEC: {dec_real}',
                        transform=ax2.transAxes,
                        ha='right',
                        va='bottom',
                        fontsize=10,
                        color='gray',
                    )



                    ax2.axis("off")
                else:
                    ax2.text(0.5, 0.5, "No se encontró imagen\nen ningún catálogo", 
                            ha="center", va="center", color="red", fontsize=14)
                    ax2.axis("off")
                    print("No se pudo recuperar una imagen de ninguna de las fuentes configuradas.")

                plt.tight_layout()
                plt.savefig(f'{archivo_corregido.replace(".fits", ".png")}')   



                survey_bool = input(f'Se encontro imagen en {encuesta['nombre']}:   ')
                if survey_bool == 'y':
                    break



    except Exception as e:
        print(f"\nOcurrió un error en la astrometría del objeto. No se guardo la imagen: {e} \n")
        archivo_fits = file
        directorio, nombre_base = os.path.split(archivo_fits)
        archivo_corregido = os.path.join(directorio, "Mirror_" + nombre_base)

        fig, ax = plt.subplots(figsize=(50, 30))

        ax.axis('off')

        # Añade el texto en la coordenada (x=0.5, y=0.5) centrado
        ax.text(0.5, 0.5, f"\nOcurrió un error en la astrometría del objeto. No se guardo la imagen: {e}", fontsize=24, color='black',
                ha='center', va='center')

        plt.savefig(f'{archivo_corregido.replace(".fits", ".png")}')


def MIRROR_CREATOR(work_dir):

    notify('MIRRORS',mode='M')

    notify('Creacion de las imagenes mirrors para hacer la comparacion con las imagenes de SDSS')

    RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')

    fits_files = [f for f in os.listdir(f"{work_dir}IMAGES/") if f.endswith(".fits") and not 'mirror'in f.lower()]
    fits_files.sort()

    fits_files_path = [os.path.join(f"{work_dir}IMAGES/", f) for f in fits_files]

    print(fits_files_path,'\n')

    for GEHR in fits_files_path:
        sky_location_helper(GEHR)




    task_verifier = task_checker(-1,RS)
    if RS['-1'][0] == True:
        notify('Ya se crearon los mirrors para la identificacion de objetos')
    elif not task_verifier:
        notify(f'No se puede correr la tarea {RS['-1'][1]} porque hay tareas anteriores pendientes.')
    else:
        try:
            notify('Ir al Jupyter notebook y revisar si funcionan esos objetos')
            RS['-1'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['-1'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)
            notify(f'X X X',mode='M')
            notify(f'Hubo un fallo en la tarea {RS['-1'][1]}')


        