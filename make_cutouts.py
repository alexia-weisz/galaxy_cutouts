import numpy as np
import config
import gal_data
import extract_stamp
import warnings
import argparse
from pdb import set_trace

#_OUT_DIR = '../cutouts/sings/'
#_MIPS_DIR = '/data/tycho/0/leroy.42/ellohess/data/mips/sings/'
#_LOS_DIR = '../../ellohess/code/index/'
_KERNEL_DIR = '/data/tycho/0/leroy.42/ellohess/kernels/Low_Resolution/'
_TEST_WRITE_DIR = '/n/home00/lewis.1590/research/galbase_allsky/cutouts/'
_GALDATA_DIR = '/n/home00/lewis.1590/research/galbase/gal_data/'

def get_args():
    """ Get command line arguments """
    parser = argparse.ArgumentParser(description='Create cutouts of a given size around each galaxy center.')
    parser.add_argument('--size', default=30, help='cutout size in arcminutes. Default: 30.')
    parser.add_argument('--band', default=None, help='waveband. Default: None (does all)')
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--model_bg', action='store_true', help='model the background to match all images as best as possible.')
    parser.add_argument('--weight_images', action='store_true', help='weight the input images by the desired weight images')
    parser.add_argument('--galaxy_list', default=None, help='Galaxy name if doing a single cutout or list of names. Default: None')
    parser.add_argument('--all_galaxies', action='store_true', help='run all galaxies in database. Default: False; include flag to store_true')
    parser.add_argument('--tag', default=None, help='tag to select galaxies, i.e., SINGS, HERACLES, etc. Default: None')
    parser.add_argument('--inds', nargs=2, type=int, help='index the all galaxy array')
    return parser.parse_args()


def main(**kwargs):
    """ Create cutouts using all data and combining images using Montage
    
    Parameters
    ----------
    size : float
        cutout size in arcminutes (Default: 30.0)
    band : str
        the band in which the cutout is made, either a single band or a list (Default: fuv)
    cutout : bool
        create a cutout
    model_bg : bool
        model the background in Montage
    weight_images : bool
        weight the input images with the weights images
    galaxy_list : list
        list of one or more galaxies for which to make cutouts. Do not set if you want to make cutouts for all galaxies (Default: None)
    all_galaxies : bool
        Make cutouts for all galaxies in the galbase
    tag : str
        A tag to select a subset of galaxies; i.e., SINGS, HERACLES, etc. (Default: None)
    inds : int
        List of two ints to index the galaxy array from [int1:int2]

    Example:
    This code can be run from the command line or imported and run within a separate program.
    The following example creates cutouts of the SINGS sample that are 30x30 arcminutes in the FUV with 
    modeled backgrounds.

    Usage:
    %run make_cutouts.py --size 30 --band fuv --cutout --model_bg --tag SINGS

    or

    import make_cutouts
    make_cutouts.main(size=30, band='fuv', cutout=True, model_bg=True, tag='SINGS')"""

    if kwargs['cutout']:
        warnings.filterwarnings('ignore')
        wband = kwargs['band']

        gals = gal_data.gal_data(names=kwargs['galaxy_list'], data=None, all=kwargs['all_galaxies'], galdata_dir=_GALDATA_DIR, tag=kwargs['tag']) #get data from galbase

        if kwargs['inds']:
            ind_start, ind_stop = kwargs['inds'][0], kwargs['inds'][1]
            gals = gals[ind_start:ind_stop]

        n_gals = len(gals)
        size_deg = kwargs['size'] * 60. / 3600. #convert from arcminutes to degrees

        for i in range(n_gals):
            #this_gal = np.rec.fromarrays(gals[i], names=list(config.COLUMNS))
            #galname = str(this_gal.name).replace(' ', '').upper()

            galname = gals['name'][0].replace(' ', '').upper()
            ra_ctr, dec_ctr = gals['ra_deg'], gals['dec_deg']

            stamp_kwargs = {'ra_ctr': ra_ctr, 'dec_ctr': dec_ctr, 'size_deg': size_deg, 'name': galname, 'model_bg': kwargs['model_bg'], 
                            'weight_images': kwargs['weight_images']}
            if wband == 'fuv':
                extract_stamp.galex(band='fuv', **stamp_kwargs)#ra_ctr=ra_ctr, dec_ctr=dec_ctr, size_deg=size_deg, name=galname, model_bg=kwargs['model_bg'])
            elif wband == 'nuv':
                extract_stamp.galex(band='nuv', **stamp_kwargs)#ra_ctr=ra_ctr, dec_ctr=dec_ctr, size_deg=size_deg, name=galname, model_bg=kwargs['model_bg'])
            else:
                extract_stamp.galex(band='fuv', **stamp_kwargs)#ra_ctr=ra_ctr, dec_ctr=dec_ctr, size_deg=size_deg, name=galname, model_bg=kwargs['model_bg'])
                extract_stamp.galex(band='nuv', **stamp_kwargs)#ra_ctr=ra_ctr, dec_ctr=dec_ctr, size_deg=size_deg, name=galname, model_bg=kwargs['model_bg'])



if __name__ == '__main__':
    args = get_args()
    main(**vars(args))
