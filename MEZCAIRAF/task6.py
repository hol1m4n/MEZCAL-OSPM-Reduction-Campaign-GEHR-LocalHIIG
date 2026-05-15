from pyraf import iraf
from pyraf import gki
from utils import leer_o_crear_logfile
from utils import notify
from utils import guardar_logfile
from utils import list_generator
from utils import Plotganizer
import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt



def ONE_DIMENTIONAL_SPECTRUM_EXTRACT(work_dir):

    notify('1D EXTR',mode='M')
    RS = leer_o_crear_logfile(f'{work_dir}/log_reduc')


    iraf.noao()
    iraf.twodspec()
    iraf.apextract()

    if RS['20'][0] == True:
        notify('Ya se realizo la conversion a 1D de los objetos calibrados. Se pueden revisar y/o combinar')
    else:
        try:
            obj_files = [f for f in os.listdir(f'{work_dir}OBJS') if f.endswith(".fits") and 'wlcal' in f.lower()]
            obj_files.sort()

            ls_tmp = [e[:-16] for e in obj_files]

            ls_unique = np.unique(ls_tmp)

            ls_match = []

            for u in ls_unique:
                c = 0
                for s in range(len(obj_files)):
                    if u in obj_files[s]:
                        c += 1
                ls_match.append(c)

            print(ls_tmp)
            print(ls_unique)
            print(ls_match)


            for spec in obj_files:

                print(f'Extraccion de aperturas para {spec}')

                iraf.apall(
                    input =f"{work_dir}OBJS/{spec}"                 , #List of input images
                    nfind =                 1, #Number of apertures to be found automatically
                    output = f"{work_dir}OBJS/{spec.replace('wlcal','1d')}"             , #List of output spectra
                    apertures = ""            , #Apertures
                    format = "multispec"    , #Extracted spectra format
                    references = ""             , #List of aperture reference images
                    profiles = ""             , #List of aperture profile images\n
                    interactive = "yes"            , #Run task interactively?
                    find = "yes"            , #Find apertures?
                    recenter = "yes"            , #Recenter apertures?
                    resize = "yes"            , #Resize apertures?
                    edit = "yes"            , #Edit apertures?
                    trace = "yes"            , #Trace apertures?
                    fittrace = "yes"            , #Fit the traced points interactively?
                    extract = "yes"            , #Extract spectra?
                    extras = "no"            , #Extract sky, sigma, etc.?
                    review = "yes"            , #Review extractions?\n
                    line = "INDEF"          , #Dispersion line
                    nsum = 500, #500             , #Number of dispersion lines to sum or median\n\n# DEFAULT APERTURE PARAMETERS\n
                    lower = -5.0            , #Lower aperture limit relative to center
                    upper = 5.0             , #Upper aperture limit relative to center
                    apidtable = ""             , #Aperture ID table (optional)\n\n# DEFAULT BACKGROUND PARAMETERS\n
                    b_function = "chebyshev"    , #Background function
                    b_order = 2              , #Background function order
                    b_sample = "-180:-100,80:170"  , #Background sample regions
                    b_naverage = -4             , #Background average or median
                    b_niterate = 1              , #Background rejection iterations
                    b_low_reject = 3.0             , #Background lower rejection sigma
                    b_high_rejec = 3.0             , #Background upper rejection sigma
                    b_grow = 0.0             , #Background rejection growing radius\n\n# APERTURE CENTERING PARAMETERS\n
                    width = 5.0             , #Profile centering width
                    radius = 10.0            , #Profile centering radius
                    threshold = 10.0             , #Detection threshold for profile centering\n\n# AUTOMATIC FINDING AND ORDERING PARAMETERS\n
                    minsep = 3.0             , #Minimum separation between spectra
                    maxsep = 100000.0        , #Maximum separation between spectra
                    order = "increasing"   , #Order of apertures\n\n# RECENTERING PARAMETERS\n
                    aprecenter = ""             , #Apertures for recentering calculation
                    npeaks = "INDEF"          , #Select brightest peaks
                    shift = "no"            , #Use average shift instead of recentering?\n\n# RESIZING PARAMETERS\n
                    llimit = "INDEF"          , #Lower aperture limit relative to center
                    ulimit = "INDEF"          , #Upper aperture limit relative to center
                    ylevel = 0.1            , #Fraction of peak or intensity for automatic width
                    peak = "yes"            , #Is ylevel a fraction of the peak?
                    bkg = "yes"            , #Subtract background in automatic width?
                    r_grow = 0.0             , #Grow limits by this factor
                    avglimits = "no"             , #Average limits over all apertures?\n\n# TRACING PARAMETERS\n
                    t_nsum = 20             , #Number of dispersion lines to sum
                    t_step = 10             , #Tracing step
                    t_nlost = 3              , #Number of consecutive times profile is lost before quitting
                    t_function = "legendre"     , #Trace fitting function
                    t_order = 3              , #Trace fitting function order
                    t_sample = "*"            , #Trace sample regions
                    t_naverage = 1              , #Trace average or median
                    t_niterate = 1              , #Trace rejection iterations
                    t_low_reject = 2.5             , #Trace lower rejection sigma
                    t_high_rejec = 2.5             , #Trace upper rejection sigma
                    t_grow = 0.0             , #Trace rejection growing radius\n\n# EXTRACTION PARAMETERS\n
                    background = "median"         , #Background to subtract
                    skybox = 1              , #Box car smoothing length for sky
                    weights = "none"         , #Extraction weights (none|variance)
                    pfit = "fit1d"        , #Profile fitting type (fit1d|fit2d)
                    clean = "no"             , #Detect and replace bad pixels?
                    saturation = "INDEF"          , #Saturation level
                    readnoise = "RDNOISE"           , #Read out noise sigma (photons)
                    gain = "GAIN"           , #Photon gain (photons/data number)
                    lsigma = 4.             , #Lower rejection threshold
                    usigma = 4.             , #Upper rejection threshold
                    nsubaps = 1              , #Number of subapertures per aperture
                )            
            RS['20'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['20'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)            
    



    notify('Ploteando espectros 1D extraidos')


    if RS['21'][0] == True:
        notify('Los espectros 1D extraidos ya estan ploteados')
    else:
        try:
            dest = os.path.join(work_dir, "Redux")
            os.makedirs(dest, exist_ok=True)
            def plot_espectro_1d(ruta_fits):
                with fits.open(ruta_fits) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header

                    if data.ndim > 1:
                        flujo = data[0] 
                    else:
                        flujo = data

                    try:
                        n_pixeles = header['NAXIS1']
                        crval = header['CRVAL1']
                        cdelt = header['CDELT1']
                        crpix = header.get('CRPIX1', 1) 
                        
                        referencia_pix = np.arange(n_pixeles) + 1
                        longitud_onda = crval + (referencia_pix - crpix) * cdelt
                        
                    except KeyError:
                        print("No se encontraron las llaves WCS. Ploteando en píxeles.")
                        longitud_onda = np.arange(len(flujo))

                    fig = plt.figure(figsize=(15, 10))
                    ax = fig.add_subplot()
                    ax.step(longitud_onda, flujo, where='mid', color='black', lw=1)
                    ax.axvline(x=6564.6, color='r', linestyle='--')
                    ax.set_xlabel(f"Longitud de Onda ({header.get('CUNIT1', 'Ang')})")
                    ax.set_ylabel("Flujo (ADUs / Electrones)")
                    ax.set_title(f"Espectro: {header.get('OBJECT', 'Sin nombre')}")
                    ax.grid(alpha=0.3)
                    fig.savefig(ruta_fits.replace('.fits', '.png'),dpi=150)
                    plt.close(fig)

            _1D = [f for f in os.listdir(f'{work_dir}OBJS') if f.endswith(".fits") and '1d' in f.lower()]
            _1D.sort()

            for spec in _1D:
                plot_espectro_1d(f'{work_dir}Redux/{spec}')

            RS['21'][0] = True
            guardar_logfile(f'{work_dir}/log_reduc',RS)
        except:
            RS['21'][0] = False
            guardar_logfile(f'{work_dir}/log_reduc',RS)             