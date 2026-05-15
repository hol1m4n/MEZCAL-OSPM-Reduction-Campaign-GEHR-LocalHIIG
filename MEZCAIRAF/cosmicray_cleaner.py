from astropy.io import fits
from lacosmic import remove_cosmics
import numpy as np

from utils import notify

def cosmicos_cleaner(line):
    line = line.strip()
    cr_file = line.replace('trim', 'crLA')
    msk_file = line.replace('trim', 'mskLA')

    with fits.open(line) as hdul_orig:
        data_ = hdul_orig[0].data
        header_ = hdul_orig[0].header

        notify(f'Removiendo cosmicos de {line}')

        if 'arc' in line.lower():
            clean_img, cr_mask = remove_cosmics(data=data_, 
                                                effective_gain=1.0, 
                                                readnoise=5.0, 
                                                maxiter=4, 
                                                cr_threshold=4.5, 
                                                neighbor_threshold=0.3, 
                                                contrast=7.0, 
                                                border_mode='mirror')
        else:
            clean_img, cr_mask = remove_cosmics(data=data_, 
                                                effective_gain=1.0, 
                                                readnoise=5.0, 
                                                maxiter=6, 
                                                cr_threshold=3.5, 
                                                neighbor_threshold=0.25, 
                                                contrast=7.0, 
                                                border_mode='mirror')

        hdu_clean = fits.PrimaryHDU(data=clean_img, header=header_)
        hdu_clean.writeto(cr_file, overwrite=True)

        hdu_mask = fits.PrimaryHDU(data=cr_mask.astype('int16'), header=header_)
        hdu_mask.writeto(msk_file, overwrite=True) 

