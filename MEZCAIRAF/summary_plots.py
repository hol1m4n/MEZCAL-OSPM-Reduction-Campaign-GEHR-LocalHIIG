import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from pathlib import Path
import os, glob
from utils import notify
from joblib import Parallel, delayed


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from pathlib import Path
import os, glob
import shutil
# ─────────────────────────────────────────────
# PASO 3 — MASTERBIAS
# Qué buscar: perfil plano, histograma angosto, sin gradientes
# ─────────────────────────────────────────────
def diag_masterbias(work_dir):
    path = f'{work_dir}BIAS/masterbias.fits'
    with fits.open(path) as hdul:
        data = hdul[0].data

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Diagnóstico Masterbias', fontsize=14)

    # 2D
    ax = axes[0, 0]
    vmin, vmax = np.percentile(data, [1, 99])
    im = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title('Imagen 2D')

    # Perfil espacial
    ax = axes[0, 1]
    ax.plot(np.median(data, axis=1), lw=0.8)
    ax.axhline(np.median(data), color='r', linestyle='--', label=f'Mediana global = {np.median(data):.2f}')
    ax.set_xlabel('Fila Y')
    ax.set_ylabel('ADUs')
    ax.set_title('Perfil espacial  [debe ser plano]')
    ax.legend()

    # Perfil espectral
    ax = axes[1, 0]
    ax.plot(np.median(data, axis=0), lw=0.8)
    ax.axhline(np.median(data), color='r', linestyle='--')
    ax.set_xlabel('Columna X')
    ax.set_ylabel('ADUs')
    ax.set_title('Perfil espectral  [debe ser plano]')

    # Histograma
    ax = axes[1, 1]
    ax.hist(data.flatten(), bins=200, color='steelblue', edgecolor='none')
    ax.set_xlabel('ADUs')
    ax.set_ylabel('N píxeles')
    mu, sigma = np.mean(data), np.std(data)
    ax.set_title(f'Histograma  μ={mu:.2f}  σ={sigma:.2f} ADUs\n'
                 f'[σ debe ser ≈ readnoise/√N_bias]')

    plt.tight_layout()
    plt.savefig(f'{work_dir}Summary/diag_masterbias.png', dpi=80)
    plt.close()
    print(f'  → Masterbias: μ={mu:.2f}, σ={sigma:.3f} ADUs')


# ─────────────────────────────────────────────
# PASO 6+7 — MASTERFLAT (crudo y normalizado)
# Qué buscar: perfil espacial plano en FlatNmaster (±2-3%)
# ─────────────────────────────────────────────
def diag_flats(work_dir):
    archivos = {
        'masterflat (crudo)':       f'{work_dir}FLATS/masterflat.fits',
        'FlatNpre_lumcor':           f'{work_dir}FLATS/FlatNpre_lumcor.fits',
        'FlatIllum':                 f'{work_dir}FLATS/FlatIllum.fits',
        'FlatNmaster (final)':       f'{work_dir}FLATS/FlatNmaster.fits',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Perfil espacial — cadena de flats', fontsize=14)

    for ax, (label, path) in zip(axes.flatten(), archivos.items()):
        if not Path(path).exists():
            ax.set_title(f'{label}\n[no encontrado]')
            continue
        with fits.open(path) as hdul:
            data = hdul[0].data
        med = np.median(data, axis=1)
        ax.plot(med, lw=0.8)
        ax.axhline(np.median(med), color='r', linestyle='--',
                   label=f'mediana={np.median(med):.3f}')
        ax.set_xlabel('Fila Y')
        ax.set_ylabel('Mediana por fila')
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{work_dir}Summary/diag_flats_perfiles.png', dpi=80)
    plt.close()

    # Histograma del FlatNmaster final
    path_fn = f'{work_dir}FLATS/FlatNmaster.fits'
    if Path(path_fn).exists():
        with fits.open(path_fn) as hdul:
            data = hdul[0].data
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('FlatNmaster final — verificación', fontsize=13)

        axes[0].hist(data.flatten(), bins=300, color='steelblue', edgecolor='none')
        axes[0].set_xlim(0.5, 1.5)
        axes[0].axvline(1.0, color='r', linestyle='--', label='Ideal = 1.0')
        axes[0].set_xlabel('Valor del píxel')
        axes[0].set_ylabel('N píxeles')
        axes[0].set_title(f'Histograma  μ={np.mean(data):.4f}  σ={np.std(data):.4f}')
        axes[0].legend()

        axes[1].plot(np.median(data, axis=0), lw=0.8)
        axes[1].axhline(1.0, color='r', linestyle='--')
        axes[1].set_xlabel('Columna X (dirección espectral)')
        axes[1].set_ylabel('Mediana por columna')
        axes[1].set_title('Perfil espectral [debe ser ~1.0]')

        plt.tight_layout()
        plt.savefig(f'{work_dir}Summary/diag_FlatNmaster_final.png', dpi=80)
        plt.close()


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Visualizacion Flats', fontsize=14)

    # Masterflat
    with fits.open(archivos['masterflat (crudo)']) as hdul:
        data = hdul[0].data
    ax = axes[0, 0] # <--- Correcto
    vmin, vmax = np.percentile(data, [1, 99])
    im = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title('Masterflat crudo')

    # FlatNpre_lumcor
    with fits.open(archivos['FlatNpre_lumcor']) as hdul:
        data = hdul[0].data
    ax = axes[0, 1] # <--- CORREGIDO (eliminada sobrescritura)
    vmin, vmax = np.percentile(data, [1, 99])
    im = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title('Flat Normalizado No Illum correction')

    # FlatIllum
    with fits.open(archivos['FlatIllum']) as hdul:
        data = hdul[0].data
    ax = axes[1, 0] # <--- CORREGIDO (eliminada sobrescritura)
    vmin, vmax = np.percentile(data, [1, 99])
    im = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title('Flat Illum correction')

    # FlatNmaster
    with fits.open(archivos['FlatNmaster (final)']) as hdul:
        data = hdul[0].data
    ax = axes[1, 1] # <--- CORREGIDO (eliminada sobrescritura)
    vmin, vmax = np.percentile(data, [1, 99])
    im = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title('FlatNmaster (final)')

    plt.tight_layout()
    plt.savefig(f'{work_dir}Summary/diag_Flats_2D_images.png', dpi=80)
    plt.close()




# ─────────────────────────────────────────────
# PASOS 11 y 15 — REMOCIÓN DE CÓSMICOS (arcos y objetos)
# Qué buscar: fracción de píxeles enmascarados (<1% arcos, <0.5% objetos)
#             Que líneas de arco NO estén enmascaradas
# ─────────────────────────────────────────────
def diag_cosmic_removal(work_dir, tipo='ARCS', sufijo_trim='trim', sufijo_cr='crLA', sufijo_msk='mskLA'):
    folder = f'{work_dir}{tipo}/'
    trim_files = sorted(glob.glob(f'{folder}*{sufijo_trim}*.fits'))

    if not trim_files:
        print(f'  No se encontraron archivos {sufijo_trim} en {folder}')
        return

    # Toma el primero como ejemplo
    def cosmic_removal_plotter(n):
        f_trim = trim_files[n]
        f_cr   = f_trim.replace(sufijo_trim, sufijo_cr)
        f_msk  = f_trim.replace(sufijo_trim, sufijo_msk)

        if not Path(f_cr).exists() or not Path(f_msk).exists():
            print(f'  Archivos crLA/mskLA no encontrados para {f_trim}')
            return

        with fits.open(f_trim) as h: orig = h[0].data
        with fits.open(f_cr)   as h: clean = h[0].data
        with fits.open(f_msk)  as h: mask = h[0].data

        frac_cr = mask.sum() / mask.size * 100
        nombre = Path(f_trim).stem

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Remoción de cósmicos — {tipo} — {nombre}', fontsize=13)

        vmin, vmax = np.percentile(orig, [1, 99])
        axes[0].imshow(orig,  cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title('Original (trim)')

        axes[1].imshow(clean, cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title('Limpia (crLA)')

        axes[2].imshow(mask,  cmap='Reds',    origin='lower', aspect='auto')
        axes[2].set_title(f'Máscara CR\n{frac_cr:.3f}% píxeles afectados\n'
                        f'[bueno si <1% arcos, <0.5% objetos]')

        plt.tight_layout()
        outname = f'{folder.replace(f'{tipo}','Summary')}diag_cosmics_{tipo.lower()}_{nombre}.png'
        plt.savefig(outname, dpi=80)
        plt.close()
        print(f'  → {tipo} CR fraction: {frac_cr:.3f}%  ({"OK" if frac_cr < 1.0 else "REVISAR"})')

    Parallel(n_jobs=-1)(delayed(cosmic_removal_plotter)(f) for f in range(len(trim_files)))        


# ─────────────────────────────────────────────
# PASO 16 — CALIBRACIÓN EN LONGITUD DE ONDA
# Qué buscar: RMS < 0.3 Å, ≥ 20 líneas identificadas por arco
# ─────────────────────────────────────────────
def diag_wl_calibration(work_dir):
    database_dir = f'{work_dir}ARCS/database/'
    db_files = sorted(glob.glob(f'{database_dir}id*'))

    if not db_files:
        print('  No se encontraron archivos en database/')
        return

    resultados = []
    for db_file in db_files:
        nombre = Path(db_file).name
        rms, nlines, function, order = None, None, None, None
        pixels, wavelengths = [], []

        with open(db_file, 'r') as f:
            lines = f.readlines()

        in_features = False
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('rms'):
                try: rms = float(line.split()[1])
                except: pass
            if line.startswith('features'):
                try: nlines = int(line.split()[1])
                except: pass
            if line.startswith('function'):
                function = line.split()[1]
            if line.startswith('order'):
                try: order = int(line.split()[1])
                except: pass
            # Leer los pares pixel-longitud de onda
            if nlines and len(pixels) < nlines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        px = float(parts[0])
                        wl = float(parts[1])
                        if 3000 < wl < 11000:  # rango plausible en Angstroms
                            pixels.append(px)
                            wavelengths.append(wl)
                    except:
                        pass

        resultados.append({
            'nombre': nombre, 'rms': rms, 'nlines': nlines,
            'function': function, 'order': order,
            'pixels': pixels, 'wavelengths': wavelengths
        })

    # Plot resumen de RMS y nlines 
    nombres = [r['nombre'][-20:] for r in resultados]
    rms_vals = [r['rms'] if r['rms'] else 0 for r in resultados]
    nlines_vals = [r['nlines'] if r['nlines'] else 0 for r in resultados]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(resultados)*0.8), 8))
    fig.suptitle('Calidad de la calibración en longitud de onda', fontsize=13)

    axes[0].bar(range(len(nombres)), rms_vals, color=['green' if r < 0.1 else 'red' for r in rms_vals])
    axes[0].axhline(0.1, color='r', linestyle='--', label='Límite recomendado 0.1 Å')
    axes[0].set_xticks(range(len(nombres)))
    axes[0].set_xticklabels(nombres, rotation=45, ha='right', fontsize=7)
    axes[0].set_ylabel('RMS (Å)')
    axes[0].set_title('RMS del ajuste por arco')
    axes[0].legend()

    axes[1].bar(range(len(nombres)), nlines_vals, color=['green' if n >= 20 else 'orange' for n in nlines_vals])
    axes[1].axhline(20, color='r', linestyle='--', label='Mínimo recomendado (20)')
    axes[1].set_xticks(range(len(nombres)))
    axes[1].set_xticklabels(nombres, rotation=45, ha='right', fontsize=7)
    axes[1].set_ylabel('N líneas identificadas')
    axes[1].set_title('Líneas identificadas por arco')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{work_dir}Summary/diag_wl_calibration_quality.png', dpi=80)
    plt.close()

    # Plot solución pixel-lambda para el primer arco con datos
    for r in resultados:
        if len(r['pixels']) > 5:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            rms_str  = f"{r['rms']:.3f} Å"  if r['rms']   is not None else "N/D"
            nlin_str = f"{r['nlines']}"     if r['nlines'] is not None else "N/D"
            fig.suptitle(f"Solución WL: {r['nombre']}  RMS={rms_str}  N={nlin_str}", fontsize=11)
            axes[0].scatter(r['pixels'], r['wavelengths'], s=20, color='steelblue')
            axes[0].set_xlabel('Píxel')
            axes[0].set_ylabel('Longitud de onda (Å)')
            axes[0].set_title('Solución pixel → λ')
            # Residuales aproximados
            if len(r['pixels']) > 3:
                coeffs = np.polyfit(r['pixels'], r['wavelengths'], 3)
                wl_fit = np.polyval(coeffs, r['pixels'])
                residuals = np.array(r['wavelengths']) - wl_fit
                axes[1].scatter(r['wavelengths'], residuals, s=20, color='steelblue')
                axes[1].axhline(0, color='r', linestyle='--')
                axes[1].set_xlabel('Longitud de onda (Å)')
                axes[1].set_ylabel('Residual (Å)')
                axes[1].set_title(f'Residuales  σ={np.std(residuals):.3f} Å')
            plt.tight_layout()
            plt.savefig(f'{work_dir}Summary/diag_wl_solucion_{r["nombre"]}.png', dpi=80)
            plt.close()
            #break


# ─────────────────────────────────────────────
# PASOS 18+19 — VERIFICACIÓN POST-TRANSFORM
# Qué buscar: líneas de arco perfectamente verticales,
#             líneas de cielo rectas en los objetos
# ─────────────────────────────────────────────
def diag_post_transform(work_dir):
    # Arcos transformados
    arcs_wlcal = sorted(glob.glob(f'{work_dir}ARCS/*wlcal*.fits'))
    objs_wlcal = sorted(glob.glob(f'{work_dir}OBJS/*wlcal*.fits'))

    def _plot_rectification(path, titulo, outpath, lineas_ref=None):


        with fits.open(path) as hdul:
            data = hdul[0].data
            header = hdul[0].header

        # Reconstruir eje de longitud de onda si existe WCS
        naxis1 = header.get('NAXIS1', data.shape[1])
        crval  = header.get('CRVAL1', 0)
        cdelt  = header.get('CDELT1', 1)
        crpix  = header.get('CRPIX1', 1)
        wl_axis = crval + (np.arange(naxis1) + 1 - crpix) * cdelt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(titulo, fontsize=12)

        vmin, vmax = np.percentile(data, [2, 98])
        axes[0].imshow(data, cmap='viridis', origin='lower', aspect='auto',
                       vmin=vmin, vmax=vmax,
                       extent=[wl_axis[0], wl_axis[-1], 0, data.shape[0]])
        axes[0].set_xlabel('Longitud de onda (Å)')
        axes[0].set_ylabel('Fila Y (espacial)')
        axes[0].set_title('Imagen 2D rectificada\n[líneas deben ser VERTICALES]')
        if lineas_ref:
            for wl, nombre in lineas_ref:
                if wl_axis[0] < wl < wl_axis[-1]:
                    axes[0].axvline(wl, color='r', linestyle='--', alpha=0.6, lw=0.8)
                    axes[0].text(wl, data.shape[0]*0.02, nombre, color='r',
                                 fontsize=6, rotation=90, va='bottom')

        # Perfil espacial
        axes[1].plot(np.median(data, axis=1))
        axes[1].set_xlabel('Fila Y')
        axes[1].set_ylabel('Mediana (ADUs)')
        axes[1].set_title('Perfil espacial\n[fondo debe ser uniforme]')

        # Espectro colapsado (1D promedio)
        axes[2].plot(wl_axis, np.median(data, axis=0), lw=0.8, color='steelblue')
        axes[2].set_xlabel('Longitud de onda (Å)')
        axes[2].set_ylabel('Flujo mediano')
        axes[2].set_title('Espectro 1D colapsado')
        if lineas_ref:
            for wl, nombre in lineas_ref:
                if wl_axis[0] < wl < wl_axis[-1]:
                    axes[2].axvline(wl, color='r', linestyle='--', alpha=0.7, lw=0.8)

        plt.tight_layout()
        plt.savefig(outpath, dpi=80)
        plt.close()

    # Líneas de cielo y arco de referencia (ThAr comunes + cielo)
    lineas_lampara = [
        (6554.1603, 'L_ThAr_Ref_A'),(6577.2146, 'L_ThAr_Ref_D'), #(6588.8756, 'L_ThAr_Ref_B') , (6569.6322, 'L_ThAr_Ref_C')
        (6583.906, 'L_ThAr_Ref_E'), (6588.5396, 'L_ThAr_Ref_F'), (6591.4845, 'L_ThAr_Ref_G'),(6593.9391, 'L_ThAr_Ref_H')
    ]
    lineas_emision = [
        (4861.33, 'Hβ'), (4958.91, '[OIII]'), (5006.84, '[OIII]'),
        (6548.05, '[NII]'), (6562.80, 'Hα'), (6583.45, '[NII]'),
        (6716.44, '[SII]'), (6730.82, '[SII]'),
    ]

    #if arcs_wlcal:
    #    _plot_rectification(
    #        arcs_wlcal[0],
    #        f'Post-Transform ARCO: {Path(arcs_wlcal[0]).name}',
    #        f'{work_dir}Summary/diag_transform_arc.png',
    #        lineas_ref=None
    #    )

    Parallel(n_jobs=-1)(
        delayed(_plot_rectification)(
            arc_path,
            f'Post-Transform ARCO: {Path(arc_path).stem}',
            f'{work_dir}Summary/diag_transform_{Path(arc_path).stem}.png',
            lineas_ref=lineas_lampara
        )
        for arc_path in arcs_wlcal
    )

    Parallel(n_jobs=-1)(
        delayed(_plot_rectification)(
            obj_path,
            f'Post-Transform OBJETO: {Path(obj_path).stem}',
            f'{work_dir}Summary/diag_transform_{Path(obj_path).stem}.png',
            lineas_ref=lineas_emision
        )
        for obj_path in objs_wlcal
    )

    '''
    for arc_path in arcs_wlcal:
        nombre = Path(arc_path).stem
        _plot_rectification(
            arc_path,
            f'Post-Transform ARCO: {nombre}',
            f'{work_dir}Summary/diag_transform_{nombre}.png',
            lineas_ref=lineas_lampara
        )



    for obj_path in objs_wlcal:
        nombre = Path(obj_path).stem
        _plot_rectification(
            obj_path,
            f'Post-Transform OBJETO: {nombre}',
            f'{work_dir}Summary/diag_transform_{nombre}.png',
            lineas_ref=lineas_emision
        )
    '''

# ─────────────────────────────────────────────
# PASO 20 — ESPECTROS 1D EXTRAÍDOS (mejorado)
# Qué buscar: líneas de emisión en las posiciones correctas,
#             continuo suave, S/N razonable
# ─────────────────────────────────────────────
def diag_1d_spectra(work_dir):
    spec_files = sorted(glob.glob(f'{work_dir}OBJS/*1d*.fits'))

    lineas = {
        'Hβ':     4861.33,
        '[OIII]': 5006.84,
        'Hα':     6562.80,
        '[NII]':  6583.45,
        '[SII]a': 6716.44,
        '[SII]b': 6730.82,
    }

    for path in spec_files:
        with fits.open(path) as hdul:
            data   = hdul[0].data
            header = hdul[0].header

        flujo = data[0] if data.ndim > 1 else data
        naxis1 = header.get('NAXIS1', len(flujo))
        crval  = header.get('CRVAL1', 0)
        cdelt  = header.get('CDELT1', 1)
        crpix  = header.get('CRPIX1', 1)
        wl = crval + (np.arange(naxis1) + 1 - crpix) * cdelt

        # Estimación simple de S/N (señal / ruido del continuo)
        continuo_mask = (wl > 5200) & (wl < 5500)
        if continuo_mask.sum() > 10:
            snr = np.median(flujo[continuo_mask]) / np.std(flujo[continuo_mask])
        else:
            snr = np.nan

        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        nombre = header.get('OBJECT', Path(path).stem)
        fig.suptitle(f'{nombre}  —  S/N continuo ≈ {snr:.1f}', fontsize=13)

        # Espectro completo
        axes[0].step(wl, flujo, where='mid', color='black', lw=0.8)
        for nombre_l, wl_l in lineas.items():
            if wl[0] < wl_l < wl[-1]:
                axes[0].axvline(wl_l, color='r', linestyle='--', alpha=0.6, lw=0.8)
                axes[0].text(wl_l, axes[0].get_ylim()[1]*0.9, nombre_l,
                             color='r', fontsize=7, rotation=90, va='top')
        axes[0].set_xlabel('Longitud de onda (Å)')
        axes[0].set_ylabel('Flujo (ADUs)')
        axes[0].set_title('Espectro completo')
        axes[0].grid(alpha=0.3)

        # Zoom en Hα+[NII]
        mask_ha = (wl > 6480) & (wl < 6750)
        if mask_ha.sum() > 5:
            axes[1].step(wl[mask_ha], flujo[mask_ha], where='mid', color='steelblue', lw=1)
            for nombre_l, wl_l in lineas.items():
                if 6480 < wl_l < 6750:
                    axes[1].axvline(wl_l, color='r', linestyle='--', alpha=0.7)
                    axes[1].text(wl_l, axes[1].get_ylim()[1]*0.9, nombre_l,
                                 color='r', fontsize=8, rotation=90, va='top')
            axes[1].set_xlabel('Longitud de onda (Å)')
            axes[1].set_ylabel('Flujo (ADUs)')
            axes[1].set_title('Zoom Hα + [NII] + [SII]')
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        outname = (path.replace('.fits', '_diag.png')).replace('OBJS','Summary')
        plt.savefig(outname, dpi=80)
        plt.close()
        print(f'  → {Path(path).name}  S/N≈{snr:.1f}')






def summary_plotter(work_dir):

    notify('PLOTS',mode='M')
    notify('DE',mode='M')
    notify('RESUMEN',mode='M')

    dest = os.path.join(work_dir, "Summary")
    os.makedirs(dest, exist_ok=True)

    diag_masterbias(work_dir)

    diag_flats(work_dir)

    diag_cosmic_removal(work_dir, tipo='ARCS', sufijo_trim='trim',
                        sufijo_cr='crLA', sufijo_msk='mskLA')

    diag_cosmic_removal(work_dir, tipo='OBJS', sufijo_trim='trim',
                        sufijo_cr='crLA', sufijo_msk='mskLA')

    diag_wl_calibration(work_dir)

    diag_post_transform(work_dir)

    #diag_1d_spectra(work_dir)

    dest = os.path.join(work_dir, "Sigma_fit")
    os.makedirs(dest, exist_ok=True)

    onedspec_redux = [f for f in os.listdir(f'{work_dir}OBJS') if f.endswith(".fits") and '_1d' in f.lower()]
    onedspec_redux.sort()

    for spec in onedspec_redux:
        shutil.copy(str(f'{work_dir}OBJS/{spec}'),str(f'{work_dir}Sigma_fit/{spec}'))

    #shutil.copy(str(f'/home/hollman/GEHR_specs/halpha_fwhm_fit.py'),str(f'{work_dir}Sigma_fit/halpha_fwhm_fit.py'))