from __future__ import print_function
import numpy as np
import astropy.io.fits
import gal_data
import config
import argparse
import os
import sys
from collections import defaultdict
from pdb import set_trace


_WORK_DIR = '/Users/lewis.1590/research/galbase'
_GALDATA_DIR = '/Users/lewis.1590/python/galbase/gal_data'
_OUTPUT_DIR = '/Users/lewis.1590/research/z0mgs'

BAND = 'fuv'


#galex_file_suffixes = ['-int', '-cnt', '-exp', '-rrhr', '-skybg', '-intbgsub', '-wt', '-flags', '-objmask', '-cat']
galex_file_suffixes = ['-int', '-intbgsub', '-rrhr']

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--galaxy_list', nargs='+', help='list of tilenames to gather')
	parser.add_argument('--output_file', default='wget_commands.sh', help='file to write commands to')
	return parser.parse_args()


def calc_tile_overlap(ra_ctr, dec_ctr, pad=0.0, min_ra=0., max_ra=180., min_dec=-90., max_dec=90.):
    """
    Find all tiles that fall within a given overlap (pad) of (ra_ctr, dec_ctr)

    Parameters
    ----------
    ra_ctr : float
        Central RA
    dec_ctr : float
        Central Dec
    pad : float, optional
        Size of region about center (Default: 0.0)
    min_ra : float. optional
        Min RA of box to search in for overlaps (Default: 0.)
    max_ra : float, optional
        Max RA of box to search in (Default 180.)
    min_dec : float, optional
        Min Dec of box to search in (Default: -90.)
    max_dec : float, optional
        Max Dec of box to search in (Default: 90.)

    Returns
    -------
    overlap : bool array
        Bool arrat indicatinng which tiles in the index file fall within the given region
    """
    overlap = ((min_dec - pad) < dec_ctr) & ((max_dec + pad) > dec_ctr)

    #TRAP HIGH LATITUDE CASE AND (I GUESS) TOSS BACK ALL TILES. DO BETTER LATER
    mean_dec = (min_dec + max_dec) * 0.5
    if np.abs(dec_ctr) + pad > 88.0:
        return overlap

    ra_pad = pad / np.cos(np.radians(mean_dec))

    # MERIDIAN CASES
    merid = np.where(max_ra < min_ra)
    overlap[merid] = overlap[merid] & ( ((min_ra-ra_pad) < ra_ctr) | ((max_ra+ra_pad) > ra_ctr) )[merid]

    # BORING CASE
    normal = np.where(max_ra > min_ra)
    overlap[normal] = overlap[normal] & ((((min_ra-ra_pad) < ra_ctr) & ((max_ra+ra_pad) > ra_ctr)))[normal]

    return overlap


def main(**kwargs):
    orig_wgetfile = os.path.join(_WORK_DIR, 'code/adam/MyAllSkyTable_akleroy.csv')
    outfile = kwargs.get('output_file', 'wget_commands.sh')
   
    with open(orig_wgetfile, 'r') as f:
        lines = f.readlines()

    galaxy_list = kwargs.get('galaxy_list', None)
    if galaxy_list is None:
        print('No galaxies requested!')
        sys.exit()

    gals = gal_data.gal_data(names=kwargs['galaxy_list'], galdata_dir=_GALDATA_DIR)

    n_gals = len(gals)
    size_deg = 30. * 60. / 3600.
    
    #tile_list = []
    for i in range(n_gals):
        this_gal = np.rec.fromarrays(gals[i], names=list(config.COLUMNS))
        galname = str(this_gal.name).replace(' ', '').upper()

        # open the index file which provides information about all the galex tiles
        indexfile = os.path.join(_OUTPUT_DIR, 'galex_index_file.fits')
        ext = 1
        index, hdr = astropy.io.fits.getdata(indexfile, ext, header=True)

        # CALCULATE TILE OVERLAP
        tile_overlaps = calc_tile_overlap(this_gal.ra_deg, this_gal.dec_deg, pad=size_deg,
                                          min_ra=index['MIN_RA'],
                                          max_ra=index['MAX_RA'],
                                          min_dec=index['MIN_DEC'],
                                          max_dec=index['MAX_DEC'])

        # FIND OVERLAPPING TILES WITH RIGHT BAND
        #  index file set up such that index['fuv'] = 1 if fuv and index['nuv'] = 1 if nuv
        ind = np.where((index[BAND]) & tile_overlaps)[0]

        # pull out the tilenames of the overlapping tiles
        infiles = index[ind]['fname']
        tile_list = [os.path.basename(infile.split('.')[0]) for infile in infiles]

        # setup the output directory and command file
        output_dir = os.path.join(_OUTPUT_DIR, galname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        newcommandfile = os.path.join(output_dir, galname + '_' + outfile)

        # in the original wget file, select out all of the filenames and paths of the retrieved files
        goodlines = lines[1:] # lines with a wget command
        allfiles = [f.split(' ')[-1].strip('"\n').split('/')[-1].split('.')[0] for f in goodlines]

        # find the locations in the allfiles list that correspond to tiles for the given galaxy
        inds = np.where(np.in1d(allfiles, tile_list))[0]
        names = np.asarray(allfiles)[inds] # just a check to make sure the tiles do match
        
        # now select out the lines with the correct tiles
        # we'll use this list to create our list of tiles to grab with various endings
        filestarts = np.asarray(goodlines)[inds] 

        # populate the wget command file by replacing -int at the end of the filename with the desired file type
        with open(newcommandfile, 'w') as g:
            for s in galex_file_suffixes:
                for f in filestarts:
                    newline = f.replace('-int', s)
                    g.writelines(newline)
    


if __name__ == '__main__':
    args = get_args()
    main(**vars(args))