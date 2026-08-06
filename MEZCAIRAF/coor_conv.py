from astropy.coordinates import SkyCoord, FK5
import astropy.units as u

# 1. Ingresa tus valores en formato sexagesimal
# RA: 'HH:MM:SS' o 'HH MM SS'
# DEC: 'DD:MM:SS' o 'DD MM SS'
ra_sex = "12:16:53.06"
dec_sex = "+36:12:14.10"

# 2. Creamos el objeto SkyCoord
# Definimos el frame como FK5 y el equinoccio en 2023.8
c_2023 = SkyCoord(
    ra=ra_sex, 
    dec=dec_sex, 
    unit=(u.hourangle, u.deg), # RA en horas, DEC en grados
    frame='fk5', 
    equinox='J2023.4'
)

# 3. Transformamos al equinoccio J2000
c_j2000 = c_2023.transform_to(FK5(equinox='J2000'))

# 4. Mostramos los resultados en formato sexagesimal para Aladin
print("--- Coordenadas Originales (2023.4) ---")
print(c_2023.to_string('hmsdms'))

print("\n--- Coordenadas Convertidas (J2000) ---")
print(c_j2000.to_string('hmsdms', sep=' ', precision=5))

# 5. Mostramos los resultados en decimales
print("\n--- Coordenadas J2000 en Decimales ---")
print(f"RA (deg): {c_j2000.ra.deg}")
print(f"DEC (deg): {c_j2000.dec.deg}")
print(f"{c_j2000.ra.deg} {c_j2000.dec.deg}")
