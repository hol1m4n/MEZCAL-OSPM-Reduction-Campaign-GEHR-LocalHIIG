#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
restHa_converter.py

Convierte uno o varios espectros FITS al sistema de reposo usando
el centro observado de la línea Halpha.

Uso típico:

    python3 restHa_converter.py *N7331* 6587

También puedes usar patrones entre comillas:

    python3 restHa_converter.py "N7331*.fits" 6587

Salidas por cada archivo:
    - copia FITS corregida al reposo:  nombre_original_rest.fits
    - archivo ASCII .dat:             nombre_original_rest.dat

El .dat tiene dos columnas:
    wavelength_rest    flux
"""

import argparse
import glob
import os
import sys
import numpy as np
from astropy.io import fits


C_KMS = 299792.458  # km/s


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convierte espectros FITS al reposo usando el centro observado de Halpha."
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Archivos FITS o patrones glob. El último argumento debe ser "
            "lambda_Ha_observada en Angstrom."
        )
    )

    parser.add_argument(
        "--ha-rest",
        type=float,
        default=6562.8,
        help="Longitud de onda de Halpha en reposo en Angstrom. Default: 6562.8"
    )

    parser.add_argument(
        "--hdu",
        type=int,
        default=0,
        help="HDU donde está el espectro. Default: 0"
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="_rest",
        help="Sufijo para los archivos de salida. Default: _rest"
    )

    return parser.parse_args()


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def expand_input_files(patterns):
    files = []

    for item in patterns:
        matches = glob.glob(item)

        if matches:
            files.extend(matches)
        else:
            if os.path.isfile(item):
                files.append(item)

    files = sorted(list(set(files)))
    return files


def get_wavelength_axis(header, n_pix):
    """
    Construye el eje de longitud de onda usando una solución lineal tipo IRAF/FITS.

    lambda_i = CRVAL1 + (i + 1 - CRPIX1) * CDELT1

    donde i empieza en 0.
    """

    if "CRVAL1" not in header:
        raise KeyError("No se encontró CRVAL1 en el header.")

    crval1 = header["CRVAL1"]
    crpix1 = header.get("CRPIX1", 1.0)

    if "CDELT1" in header:
        cdelt1 = header["CDELT1"]
        cdelt_keyword = "CDELT1"
    elif "CD1_1" in header:
        cdelt1 = header["CD1_1"]
        cdelt_keyword = "CD1_1"
    else:
        raise KeyError("No se encontró CDELT1 ni CD1_1 en el header.")

    pix = np.arange(n_pix, dtype=float)
    wavelength = crval1 + (pix + 1.0 - crpix1) * cdelt1

    return wavelength, crval1, cdelt1, crpix1, cdelt_keyword


def correct_header_to_rest(header, z, cdelt_keyword):
    """
    Corrige CRVAL1 y CDELT1/CD1_1 dividiendo entre 1+z.

    Esto conserva el mismo número de pixeles, pero cambia la calibración
    de longitud de onda al sistema de reposo.
    """

    factor = 1.0 + z

    header["CRVAL1"] = header["CRVAL1"] / factor

    if cdelt_keyword in header:
        header[cdelt_keyword] = header[cdelt_keyword] / factor

    if "CDELT1" in header and cdelt_keyword != "CDELT1":
        header["CDELT1"] = header["CDELT1"] / factor

    if "CD1_1" in header and cdelt_keyword != "CD1_1":
        header["CD1_1"] = header["CD1_1"] / factor

    return header


def relativistic_velocity_from_wavelength(lambda_obs, lambda_rest):
    """
    Velocidad radial relativista equivalente al corrimiento observado.

    beta = [(lambda_obs/lambda_rest)^2 - 1] /
           [(lambda_obs/lambda_rest)^2 + 1]

    v = beta c
    """

    ratio = lambda_obs / lambda_rest
    beta = (ratio**2 - 1.0) / (ratio**2 + 1.0)
    return C_KMS * beta


def process_spectrum(filename, lambda_ha_obs, lambda_ha_rest, hdu_index, suffix):
    z = lambda_ha_obs / lambda_ha_rest - 1.0
    velocity_classical = C_KMS * z
    velocity_relativistic = relativistic_velocity_from_wavelength(
        lambda_ha_obs, lambda_ha_rest
    )

    base, ext = os.path.splitext(filename)
    out_fits = f"{base}{suffix}.fits"
    out_dat = f"{base}{suffix}.dat"

    with fits.open(filename) as hdul:
        hdu = hdul[hdu_index]
        data = hdu.data
        header = hdu.header

        if data is None:
            raise ValueError(f"El HDU {hdu_index} no contiene datos.")

        data = np.asarray(data)

        # Caso típico: espectro 1D
        if data.ndim == 1:
            flux = data.astype(float)
            n_pix = flux.size

        # Caso común alternativo: imagen 2D con una sola fila o columna útil
        elif data.ndim == 2:
            if data.shape[0] == 1:
                flux = data[0, :].astype(float)
                n_pix = flux.size
            elif data.shape[1] == 1:
                flux = data[:, 0].astype(float)
                n_pix = flux.size
            else:
                raise ValueError(
                    f"{filename}: el HDU contiene datos 2D con forma {data.shape}. "
                    "Este script espera un espectro 1D, o una imagen 2D de una sola fila/columna."
                )
        else:
            raise ValueError(
                f"{filename}: datos con ndim={data.ndim}. Este script espera un espectro 1D."
            )

        wave_obs, crval1, cdelt1, crpix1, cdelt_keyword = get_wavelength_axis(
            header, n_pix
        )

        wave_rest = wave_obs / (1.0 + z)

        # Crear copia del FITS original
        hdul_out = fits.HDUList([hdu.copy() if i == hdu_index else hdul[i].copy()
                                 for i in range(len(hdul))])

        out_header = hdul_out[hdu_index].header

        # Corregir WCS espectral en el header
        out_header = correct_header_to_rest(out_header, z, cdelt_keyword)

        # Agregar información nueva al header
        out_header["HAOBS"] = (
            float(lambda_ha_obs),
            "Observed Halpha center before rest correction [Angstrom]"
        )
        out_header["HAREST"] = (
            float(lambda_ha_rest),
            "Rest-frame Halpha wavelength used [Angstrom]"
        )
        out_header["Z_HA"] = (
            float(z),
            "Redshift inferred from observed Halpha"
        )
        out_header["VHA_KMS"] = (
            float(velocity_relativistic),
            "Relativistic velocity from Halpha shift [km/s]"
        )
        out_header["VHA_CL"] = (
            float(velocity_classical),
            "Classical velocity c*z from Halpha shift [km/s]"
        )
        out_header["RESTCOR"] = (
            True,
            "Spectrum wavelength axis corrected to rest frame"
        )

        # Guardar FITS corregido
        hdul_out.writeto(out_fits, overwrite=True)

    # Guardar archivo .dat
    dat_array = np.column_stack([wave_rest, flux])

    header_dat = (
        f"# Rest-frame spectrum generated from: {filename}\n"
        f"# lambda_Ha_obs = {lambda_ha_obs:.8f} Angstrom\n"
        f"# lambda_Ha_rest = {lambda_ha_rest:.8f} Angstrom\n"
        f"# z_Ha = {z:.10e}\n"
        f"# v_Ha_relativistic = {velocity_relativistic:.6f} km/s\n"
        f"# v_Ha_classical = {velocity_classical:.6f} km/s\n"
        f"# columns: wavelength_rest[Angstrom] flux_or_counts\n"
    )

    np.savetxt(
        out_dat,
        dat_array,
        fmt="%.8f %.8e",
        header=header_dat,
        comments=""
    )

    return out_fits, out_dat, z, velocity_relativistic, velocity_classical


def main():
    args = parse_arguments()

    # El último argumento debe ser lambda_Ha_obs
    if not is_float(args.inputs[-1]):
        print(
            "ERROR: El último argumento debe ser la longitud de onda observada "
            "de Halpha en Angstrom.",
            file=sys.stderr
        )
        print(
            "Ejemplo: python3 restHa_converter.py *N7331* 6587",
            file=sys.stderr
        )
        sys.exit(1)

    lambda_ha_obs = float(args.inputs[-1])
    file_patterns = args.inputs[:-1]

    if len(file_patterns) == 0:
        print("ERROR: No se proporcionaron archivos o patrones FITS.", file=sys.stderr)
        sys.exit(1)

    files = expand_input_files(file_patterns)

    if len(files) == 0:
        print("ERROR: No se encontraron archivos que coincidan con el patrón.", file=sys.stderr)
        sys.exit(1)

    print("")
    print("==============================================")
    print(" Conversión de espectros al reposo usando Halpha")
    print("==============================================")
    print(f"Halpha observada : {lambda_ha_obs:.6f} Angstrom")
    print(f"Halpha reposo    : {args.ha_rest:.6f} Angstrom")
    print(f"N archivos       : {len(files)}")
    print("")

    for filename in files:
        try:
            out_fits, out_dat, z, vrel, vcl = process_spectrum(
                filename=filename,
                lambda_ha_obs=lambda_ha_obs,
                lambda_ha_rest=args.ha_rest,
                hdu_index=args.hdu,
                suffix=args.suffix
            )

            print(f"[OK] {filename}")
            print(f"     z_Ha      = {z:.8e}")
            print(f"     v_rel     = {vrel:.3f} km/s")
            print(f"     v_classic = {vcl:.3f} km/s")
            print(f"     FITS out  = {out_fits}")
            print(f"     DAT out   = {out_dat}")
            print("")

        except Exception as e:
            print(f"[ERROR] {filename}: {e}", file=sys.stderr)
            print("")


if __name__ == "__main__":
    main()