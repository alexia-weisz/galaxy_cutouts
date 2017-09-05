import numpy as np
import astropy.io.fits as pyfits
import config
from pdb import set_trace
import gal_data
import extract_stamp
import warnings

_OUT_DIR = '../cutouts/sings/'
_MIPS_DIR = '/data/tycho/0/leroy.42/ellohess/data/mips/sings/'
_LOS_DIR = '../../ellohess/code/index/'
_KERNEL_DIR = '/data/tycho/0/leroy.42/ellohess/kernels/Low_Resolution/'
_TEST_WRITE_DIR = '/n/home00/lewis.1590/research/galbase_allsky/cutouts/'
_GALDATA_DIR = '/n/home00/lewis.1590/research/galbase/gal_data/'

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create cutouts of a given size around each galaxy center.')
    parser.add_argument('--size', default=30, help='cutout size in arcminutes. Default: 30.')
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--band', default='fuv', help='waveband. Default: fuv')
    parser.add_argument('--copy', action='store_true')
    parser.add_argument('--convolve', action='store_true')
    parser.add_argument('--align', action='store_true')
    parser.add_argument('--model_bg', action='store_true', help='model the background to match all images as best as possible.')
    parser.add_argument('--galaxy_list', default=None, help='Galaxy name if doing a single cutout or list of names. Default: None')
    parser.add_argument('--all_galaxies', default=False, help='run all galaxies in database. Default: False')
    parser.add_argument('--tag', default=None, help='tag to select galaxies, i.e., SINGS, HERACLES, etc. Default: None')
    return parser.parse_args()


def main(**kwargs):

    if kwargs['cutout']:
        warnings.filterwarnings('ignore')
        wband = kwargs['band']

        gals = gal_data.gal_data(names=kwargs['galaxy_list'], data=None, all=kwargs['all_galaxies'], galdata_dir=_GALDATA_DIR, tag=kwargs['tag'])
        n_gals = len(gals)
        size_deg = kwargs['size'] * 60. / 3600.

        for i in range(n_gals):
            this_gal = np.rec.fromarrays(gals[i], names=list(config.COLUMNS))
            galname = str(this_gal.name).replace(' ', '').upper()

            if wband == 'fuv':
                extract_stamp.galex(band='fuv', ra_ctr=this_gal.ra_deg, dec_ctr=this_gal.dec_deg, size_deg=size_deg, name=galname, model_bg=kwargs['model_bg'])
            elif wband == 'nuv':
                extract_stamp.galex(band='nuv', ra_ctr=this_gal.ra_deg, dec_ctr=this_gal.dec_deg, size_deg=size_deg, name=galname, model_bg=kwargs['model_bg'])
            else:
                extract_stamp.galex(band='fuv', ra_ctr=this_gal.ra_deg, dec_ctr=this_gal.dec_deg, size_deg=size_deg, name=galname, model_bg=kwargs['model_bg'])
                extract_stamp.galex(band='nuv', ra_ctr=this_gal.ra_deg, dec_ctr=this_gal.dec_deg, size_deg=size_deg, name=galname, model_bg=kwargs['model_bg'])

    if kwargs['copy']:
        pass

    if kwargs['convolve']:
        pass

    if kwargs['align']:
        pass



if __name__ == '__main__':
    args = get_args()
    main(**vars(args))
