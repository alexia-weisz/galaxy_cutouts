import numpy as np
import astropy.io.fits
import astropy.wcs
import montage_wrapper as montage
from matplotlib.path import Path
import os
import sys
import shutil
import glob
import time
import argparse
import warnings
import gal_data
import scipy.ndimage as sp

from pdb import set_trace



_GALDATA_DIR = '/n/home00/lewis.1590/research/galbase/gal_data/'


# directories that contain the input data
_TOP_DIR = '/data/tycho/0/leroy.42/allsky/'
_INDEX_DIR = os.path.join(_TOP_DIR, 'z0mgs/')
_WISE_DIR = os.path.join(_TOP_DIR, 'unwise', 'atlas')

# directories to do the work in
_WORK_DIR = '/data/tycho/0/leroy.42/allsky/galex/atlas'
#_WORK_DIR = '/data/tycho/0/lewis.1590/atlas/'
_MOSAIC_DIR = os.path.join(_WORK_DIR, 'cutouts')

# CALIBRATION FROM GALEX COUNTS TO ABMAG
FUV2AB = 18.82
NUV2AB = 20.08
UV2AB = {'fuv': FUV2AB, 'nuv': NUV2AB}

GALEX_PIX_AS = 1.5 ## galex pixel scale in arcseconds -- from documentation

# window size for creating noise mosaic
WINDOW_SIZE = 3


class GalaxyHeader(object):
    def __init__(self, name, gal_dir, ra_ctr, dec_ctr, pix_len, pix_scale, factor=1):
        self.name = name
        self.gal_dir = gal_dir
        self.ra_ctr = ra_ctr
        self.dec_ctr = dec_ctr
        self.hdr, self.hdrfile = self._create_hdr_output(pix_len, pix_scale, factor=1)
        self.hdr_ext, self.hdrfile_ext = self._create_hdr_output(pix_len, pix_scale, factor=factor)

    def _create_hdr_obj(self, pix_len, pix_scale):
        """
        Create a FITS header

        Parameters
        ----------
        ra_ctr : float
            RA of center of galaxy
        dec_ctr : float
            Dec of center of galaxy
        pix_len : float
            Length of each axis (square, so the same for x and y)
        pix_scale : float
            Pixel scale in degrees

        Returns
        -------
        hdr : astropy Header() object
            Newly created header object
        """
        hdr = astropy.io.fits.Header()
        hdr['NAXIS'] = 2
        hdr['NAXIS1'] = pix_len
        hdr['NAXIS2'] = pix_len
        hdr['CTYPE1'] = 'RA---TAN'
        hdr['CRVAL1'] = float(self.ra_ctr)
        hdr['CRPIX1'] = (pix_len / 2.) * 1.
        hdr['CDELT1'] = -1.0 * pix_scale
        hdr['CTYPE2'] = 'DEC--TAN'
        hdr['CRVAL2'] = float(self.dec_ctr)
        hdr['CRPIX2'] = (pix_len / 2.) * 1.
        hdr['CDELT2'] = pix_scale
        hdr['EQUINOX'] = 2000
        return hdr


    def _create_hdr_output(self, size_degrees, pixel_scale, factor=1):
        """
        Create a header and write it to an ascii file for use in Montage

        Parameters
        ----------
        galname : str
            Name of the galaxy
        ra_ctr : float
            Central RA of galaxy
        dec_ctr : float
            Central Dec of galaxy
        size_degrees : float
            size of cutout, in degrees
        pixel_scale : float
            pixel scale of output in arcseconds per pixel
        factor : int, optional
            Number by which to multiply size_degrees to extend the size of the cutout for bg modeling. (Default: 1)

        Returns
        -------
        target_hdr : astropy.header object
            The output header object
        header_file : str
            Path to the ascii file containing the header information
        """
        pix_len = int(np.ceil(size_degrees * factor / pixel_scale))
        hdr = self._create_hdr_obj(pix_len, pixel_scale)
        ri_targ, di_targ = self._make_axes(hdr)
        sz_out = ri_targ.shape
        outim = ri_targ * np.nan

        prihdu = astropy.io.fits.PrimaryHDU(data=outim, header=hdr)
        target_hdr = prihdu.header

        suff = '_template.hdr'
        if factor != 1:
            suff = suff.replace('.hdr', '_ext.hdr')
        header_file = os.path.join(self.gal_dir, self.name + suff)
        self.write_headerfile(header_file, target_hdr)

        return target_hdr, header_file


    def _make_axes(self, hdr, quiet=False, novec=False, vonly=False, simple=False):
        """
        Create axes arrays for the new mosaiced image. This is a simple translation to Python of Adam's
        IDL routine of the same name.

        Parameters
        ----------
        hdr : FITS header object
            FITS header to hold astrometry of desired output image
        quiet : bool, optional
            NOT USED
        novec : bool
            Find RA and Dec for every point (Default: False)
        vonly : bool
            Return only velocity data (Default: False)
        simple : bool
            Do the simplest thing (Default: False)

        Returns
        -------
        rimg : array
            array for ouptut RA
        dimg : array
            array for output Dec
        """

        # PULL THE IMAGE/CUBE SIZES FROM THE HEADER
        naxis  = int(hdr['NAXIS'])
        naxis1 = int(hdr['NAXIS1'])
        naxis2 = int(hdr['NAXIS2'])
        if naxis > 2:
            naxis3 = hdr['NAXIS3']

        ## EXTRACT FITS ASTROMETRY STRUCTURE
        ww = astropy.wcs.WCS(hdr)

        #IF DATASET IS A CUBE THEN WE MAKE THE THIRD AXIS IN THE SIMPLEST WAY POSSIBLE (NO COMPLICATED ASTROMETRY WORRIES FOR FREQUENCY INFORMATION)
        if naxis > 3:
            #GRAB THE RELEVANT INFORMATION FROM THE ASTROMETRY HEADER
            cd = ww.wcs.cd
            crpix = ww.wcs.crpix
            cdelt = ww.wcs.crelt
            crval = ww.wcs.crval

        if naxis > 2:
        # MAKE THE VELOCITY AXIS (WILL BE M/S)
            v = np.arange(naxis3) * 1.0
            vdif = v - (hdr['CRPIX3']-1)
            vaxis = (vdif * hdr['CDELT3'] + hdr['CRVAL3'])

        # CUT OUT HERE IF WE ONLY WANT VELOCITY INFO
        if vonly:
            return vaxis

        #IF 'SIMPLE' IS CALLED THEN DO THE REALLY TRIVIAL THING:
        if simple:
            print('Using simple aproach to make axes.')
            print('BE SURE THIS IS WHAT YOU WANT! It probably is not.')
            raxis = np.arange(naxis1) * 1.0
            rdif = raxis - (hdr['CRPIX1'] - 1)
            raxis = (rdif * hdr['CDELT1'] + hdr['CRVAL1'])

            daxis = np.arange(naxis2) * 1.0
            ddif = daxis - (hdr['CRPIX1'] - 1)
            daxis = (ddif * hdr['CDELT1'] + hdr['CRVAL1'])

            rimg = raxis # (fltarr(naxis2) + 1.)
            dimg = (np.asarray(naxis1) + 1.) # daxis
            return rimg, dimg

        # OBNOXIOUS SFL/GLS THING
        glspos = ww.wcs.ctype[0].find('GLS')
        if glspos != -1:
            ctstr = ww.wcs.ctype[0]
            newtype = 'SFL'
            ctstr.replace('GLS', 'SFL')
            ww.wcs.ctype[0] = ctstr
            print('Replaced GLS with SFL; CTYPE1 now =' + ww.wcs.ctype[0])

        glspos = ww.wcs.ctype[1].find('GLS')
        if glspos != -1:
            ctstr = ww.wcs.ctype[1]
            newtype = 'SFL'
            ctstr.replace('GLS', 'SFL')
            ww.wcs.ctype[1] = ctstr
            print('Replaced GLS with SFL; CTYPE2 now = ' + ww.wcs.ctype[1])

        # CALL 'xy2ad' TO FIND THE RA AND DEC FOR EVERY POINT IN THE IMAGE
        if novec:
            rimg = np.zeros((naxis1, naxis2))
            dimg = np.zeros((naxis1, naxis2))
            for i in range(naxis1):
                j = np.asarray([0 for i in xrange(naxis2)])

                pixcrd = np.array([[zip(float(i), float(j))]], numpy.float_)
                ra, dec = ww.all_pix2world(pixcrd, 1)

                rimg[i, :] = ra
                dimg[i, :] = dec
        else:
            ximg = np.arange(naxis1) * 1.0
            yimg = np.arange(naxis1) * 1.0
            X, Y = np.meshgrid(ximg, yimg, indexing='xy')
            ss = X.shape
            xx, yy = X.flatten(), Y.flatten()

            pixcrd = np.array(zip(xx, yy), np.float_)
            img_new = ww.all_pix2world(pixcrd, 0)
            rimg_new, dimg_new = img_new[:,0], img_new[:,1]

            rimg = rimg_new.reshape(ss)
            dimg = dimg_new.reshape(ss)

        # GET AXES FROM THE IMAGES. USE THE CENTRAL COLUMN AND CENTRAL ROW
        raxis = np.squeeze(rimg[:, naxis2/2])
        daxis = np.squeeze(dimg[naxis1/2, :])

        return rimg, dimg


    def write_headerfile(self, header_file, header):
        """
        Write out the header for the output mosaiced image

        Parameters
        ----------
        header_file : str
            Path to file to which to write header
        header : array
            The header to which to write to ASCII file
        """
        f = open(header_file, 'w')
        for iii in range(len(header)):
            outline = str(header[iii:iii+1]).strip().rstrip('END').strip()+'\n'
            f.write(outline)
        f.close()


    def append2hdr(self, keyword=None, value=None, ext=False):
        """
        Append information to the header and write to ASCII file

        Parameters
        ----------
        headerfile : str
            The path to the ascii file containing the header information
        keyword : str, optional
            The keyword in the header that you want to create (Default: None)
        value : multiple, optional
            The value to apply to the keyword (Default: None)
        """
        if keyword is not None:
            if ext:
                self.hdr_ext[keyword] = value
                self.write_headerfile(self.hdrfile_ext, self.hdr_ext)
            else:
                self.hdr[keyword] = value
                self.write_headerfile(self.hdrfile, self.hdr)


def make_mosaic(band='fuv', ra_ctr=None, dec_ctr=None, size_deg=None, index=None, name=None, pgcname=None, model_bg=True, weight_ims=True, convert_mjysr=True, desired_pix_scale=GALEX_PIX_AS, imtype='intbgsub', wttype='rrhr', window=False):
    """
    Create noise of a galaxy in a single GALEX band.

    Parameters
    ----------
    band : str
        GALEX band to use
    ra_ctr : float
        Central RA of galaxy
    dec_ctr : float
        Central Dec of galaxy
    size_deg : float
        Desired side length of each cutout, in degrees
    index : array, optional
        Structured array containing the galbase information. The default is to read it in inside this code. (Default: None)
    name : str, optional
        Name of the galaxy for which to generate a cutout
    pgcname : str, optional
        PGC name of the galaxy
    model_bg : bool, optional
        Model the background of the mosaiced image (Default: False)
    weight_ims : bool, optional
         weight the input images with the weights images
    convert_mjysr : bool, optional
        convert input images from counts/sec to MJy/sr
    desired_pix_scale : float, optional
        Desired pixel scale of output image. Default is currently set to GALEX pixel scale (Default: 1.5)
    imtype : str, optional
        input image type to use from galex (Default: int)
    wttype : str, optional
        input weights image type to use from galex (Default: rrhr)
    window : bool, optional
        window across the input images rather than use a single value
    """
    ttype = 'galex'
    data_dir = os.path.join(_TOP_DIR, ttype, 'sorted_tiles')
    problem_file = os.path.join(_WORK_DIR, 'problem_galaxies_{}_noise.txt'.format(band))
    numbers_file = os.path.join(_WORK_DIR, 'gal_reproj_info_{}_noise.txt'.format(band))

    galaxy_noise_file = os.path.join(_MOSAIC_DIR, '_'.join([pgcname, band]).upper() + '_noise.fits')

    if not os.path.exists(galaxy_noise_file):
        start_time = time.time()
        print pgcname, band.upper()

        # READ THE INDEX FILE (IF NOT PASSED IN)
        if index is None:
            indexfile = os.path.join(_INDEX_DIR, 'galex_index_file.fits')
            ext = 1
            index, hdr = astropy.io.fits.getdata(indexfile, ext, header=True)

        # CALCULATE TILE OVERLAP
        tile_overlaps = calc_tile_overlap(ra_ctr, dec_ctr, pad=size_deg,
                                          min_ra=index['MIN_RA'],
                                          max_ra=index['MAX_RA'],
                                          min_dec=index['MIN_DEC'],
                                          max_dec=index['MAX_DEC'])

        # FIND OVERLAPPING TILES WITH RIGHT BAND
        #  index file set up such that index['fuv'] = 1 where fuv and
        #                              index['nuv'] = 1 where nuv
        ind = np.where((index[band]) & tile_overlaps)


        # MAKE SURE THERE ARE OVERLAPPING TILES
        ct_overlap = len(ind[0])
        if ct_overlap == 0:
            with open(problem_file, 'a') as myfile:
                myfile.write(pgcname + ': ' + 'No overlapping tiles\n')
            return

        pix_scale = desired_pix_scale / 3600.  # 1.5 arbitrary: how should I set it?

        try:
            # CREATE NEW TEMP DIRECTORY TO STORE TEMPORARY FILES
            gal_dir = os.path.join(_WORK_DIR, '_'.join([pgcname, band]).upper())
            os.makedirs(gal_dir)

            # MAKE HEADER AND EXTENDED HEADER AND WRITE TO FILE
            gal_hdr = GalaxyHeader(pgcname, gal_dir, ra_ctr, dec_ctr, size_deg, pix_scale, factor=3)


            # GATHER THE INPUT FILES
            input_dir = os.path.join(gal_dir, 'input')
            if not os.path.exists(input_dir):
                os.makedirs(input_dir)
            nfiles = get_input(index, ind, data_dir, input_dir, hdr=gal_hdr)
            im_dir, wt_dir = input_dir, input_dir

            # WRITE TABLE OF INPUT IMAGE INFORMATION
            input_table = os.path.join(im_dir, 'input.tbl')
            montage.mImgtbl(im_dir, input_table, corners=True)
           
            if convert_mjysr:
                converted_dir = os.path.join(gal_dir, 'converted')
                if not os.path.exists(converted_dir):
                    os.makedirs(converted_dir)
                convert_to_flux_input(im_dir, converted_dir, band, desired_pix_scale, imtype=imtype)
                im_dir = converted_dir


            # MASK IMAGES
            masked_dir = os.path.join(gal_dir, 'masked')
            im_masked_dir = os.path.join(masked_dir, imtype)
            wt_masked_dir = os.path.join(masked_dir, wttype)
            for outdir in [masked_dir, im_masked_dir, wt_masked_dir]:
                os.makedirs(outdir)

            mask_images(im_dir, wt_dir, im_masked_dir, wt_masked_dir, imtype=imtype, wttype=wttype)
            im_dir = im_masked_dir
            wt_dir = wt_masked_dir


            # CREATE DIRECTORY FOR NOISE IMAGES
            noise_dir = os.path.join(gal_dir, 'noise')
            if not os.path.exists(noise_dir):
                os.makedirs(noise_dir)


            # CALCULATE NOISE AND GENERATE NOISE MOSAIC CUTOUT
            noisetype = 'noise'
            calc_noise(gal_dir, noise_dir, gal_hdr, galaxy_noise_file, imtype, wttype, noisetype, window=window)


            # REMOVE TEMP GALAXY DIRECTORY AND EXTRA FILES
            shutil.rmtree(gal_dir, ignore_errors=True)


            # NOTE TIME TO FINISH
            stop_time = time.time()
            total_time = (stop_time - start_time) / 60.


            # WRITE OUT THE NUMBER OF TILES THAT OVERLAP THE GIVEN GALAXY
            out_arr = [pgcname, band.upper(), nfiles, np.around(total_time, 2)]
            with open(numbers_file, 'a') as nfile:
                nfile.write('{0: >10}'.format(out_arr[0]))
                nfile.write('{0: >6}'.format(out_arr[1]))
                nfile.write('{0: >6}'.format(out_arr[2]))
                nfile.write('{0: >6}'.format(out_arr[3]) + '\n')


        # SOMETHING WENT WRONG -- WRITE ERROR TO FILE
        except Exception as inst:
            me = sys.exc_info()[0]
            with open(problem_file, 'a') as myfile:
                myfile.write(pgcname + ': ' + str(me) + ': '+str(inst)+'\n')
            shutil.rmtree(gal_dir, ignore_errors=True)

    return


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


def get_input(index, ind, data_dir, input_dir, hdr=None):
    """
    Gather the input files for creating mosaics and copy them into a temporary directory

    Parameters
    ----------
    index : np.array
        structured array from galbase FITS file
    ind : np.array(np.dtype(int))
        An array of indices into the index locating the correct files
    data_dir : str
        Path to location of raw data downloaded from GALEX (or other) server
    input_dir : str
        Path to newly created temporary directory for storing temp files used in mosaicing

    Returns
    -------
    len(input_files) : int
        The number of files that will go into the mosaic.
    """
    infiles = index[ind[0]]['fname']
    wtfiles = index[ind[0]]['rrhrfile']
    flgfiles = index[ind[0]]['flagfile']
    infiles = [os.path.join(data_dir, f) for f in infiles]
    wtfiles = [os.path.join(data_dir, f) for f in wtfiles]
    flgfiles = [os.path.join(data_dir, f) for f in flgfiles]

    for i, infile in enumerate(infiles):
        basename = os.path.basename(infile)
        new_in_file = os.path.join(input_dir, basename)
        os.symlink(infile, new_in_file)
        if hdr is not None:
            keyw = 'INFILE{}'.format(str(i+1).zfill(2))
            hdr.append2hdr(keyword=keyw, value=basename, ext=False)

    for wtfile in wtfiles:
        basename = os.path.basename(wtfile)
        new_wt_file = os.path.join(input_dir, basename)
        os.symlink(wtfile, new_wt_file)

    for flgfile in flgfiles:
        basename = os.path.basename(flgfile)
        new_flg_file = os.path.join(input_dir, basename)
        os.symlink(flgfile, new_flg_file)

    return len(infiles)
        

def convert_to_flux_input(indir, outdir, band, pix_as, imtype='intbgsub'):
    infiles = sorted(glob.glob(os.path.join(indir, '*-{}.fits'.format(imtype))))
    for infile in infiles:
        data, hdr = astropy.io.fits.getdata(infile, header=True)
        newdata = counts2jy_galex(data, UV2AB[band.lower()], pix_as)
        hdr['BUNIT'] = 'MJY/SR'
        outfile = os.path.join(outdir, os.path.basename(infile))
        astropy.io.fits.writeto(outfile, newdata, hdr)


def counts2jy_galex(counts, cal, pix_as):
    """
    Convert GALEX counts/s to MJy/sr

    Parameters
    ----------
    counts : float
        Array containing counts data to be converted
    cal : float
        Calibration value from counts to AB mag for desired band (FUV or NUV)
    pix_as : float
        Pixel scale in arcseconds

    Returns
    -------
    val : float
        Count rate converted to MJy/sr
    """
    # first convert to abmag
    abmag = -2.5 * np.log10(counts) + cal

    # then convert to Jy
    f_nu = 10**(abmag/-2.5) * 3631.

    # then to MJy
    f_nu *= 1e-6

    # then to MJy/sr
    pix_rad = np.radians(pix_as / 3600.) # pixel scale coverted from arcsec to radians
    pix_sr = pix_rad ** 2. # pixel scale converted from radians to steradians
    val = f_nu / pix_sr

    return val


def mask_images(im_dir, wt_dir, im_masked_dir, wt_masked_dir, imtype='intbgsub', wttype='rrhr'):
    """
    Mask pixels in the input images

    Parameters
    ----------
    im_dir : str
        Path to directory containing the images
    wt_dir : str
        Path to directory containing the weights
    im_masked_dir : str
        Path to temp directory for this galaxy in which to store masked image files
    wt_masked_dir : str
        Path to temp directory for this galaxy in which to store masked weight files
    """
    int_suff, rrhr_suff = '*-{}.fits'.format(imtype), '*-{}.fits'.format(wttype)
    int_images = sorted(glob.glob(os.path.join(im_dir, int_suff)))
    rrhr_images = sorted(glob.glob(os.path.join(wt_dir, rrhr_suff)))

    for i in range(len(int_images)):
        image_infile = int_images[i]
        wt_infile = rrhr_images[i]

        image_outfile = os.path.join(im_masked_dir, os.path.basename(image_infile))
        wt_outfile = os.path.join(wt_masked_dir, os.path.basename(wt_infile))

        mask_galex(image_infile, wt_infile, image_outfile, wt_outfile)


def mask_galex(intfile, wtfile, out_intfile, out_wtfile, chip_rad=1400, chip_x0=1920, chip_y0=1920):
    """
    The actual masking routine. Selects pixels that are close to the edges of the chips
    or that have bad values, and masks them.

    Parameters
    ----------
    intfile : str
        input image file
    wtfile : str
        input weight file
    chip_rad : int
        Radius of the GALEX chip to use. The actual radius of data is ~1500 pixels. There are known edge effects. (Default: 1400)
    chip_x0 : int
        Center of GALEX chip on the x-axis (Default: 1920)
    chip_y0 : int
        Center of GALEX chip on the y-axis (Default: 1920)
    out_intfile : str, optional
        Path to output, masked image file. If not included, it will default to replacing the input file name as
        '.fits' --> '_masked.fits' (Default: None)
    out_wtfile : str, optional
        Path to output, masked weight file. If not included, it will default to replacing the input file name as
        '.fits' --> '_masked.fits' (Default: None)
    """

    if not os.path.exists(out_intfile):
        # read in the data
        data, hdr = astropy.io.fits.getdata(intfile, header=True)
        wt, whdr = astropy.io.fits.getdata(wtfile, header=True)

        # determine distance of each pixel from the center
        x = np.arange(data.shape[1]).reshape(1, -1) + 1
        y = np.arange(data.shape[0]).reshape(-1, 1) + 1
        r = np.sqrt((x - chip_x0)**2 + (y - chip_y0)**2)

        # make pixel selections for masking
        i = (r > chip_rad)
        j = (wt == -1.1e30)

        # mask pixels that meet eithr of the above criterion
        data = np.where(i | j, np.nan, data)  #0
        wt = np.where(i | j, np.nan, wt) #1e-20

        # write new data to file
        astropy.io.fits.writeto(out_intfile, data, hdr)
        astropy.io.fits.writeto(out_wtfile, wt, whdr)


def calc_noise(gal_dir, this_noise_dir, gal_hdr, mosaic_file, imtype, wttype, noisetype, window=False):
    """
    Calculate noise values and generate noise mosaic for each galaxy

    Parameters:
    -----------
    gal_dir : str path
        Path to temporary directory in which mosaic work is being done
    this_noise_dir : str path
        Path to directory within gal_dir where the noise work is completed
    gal_hdr : ascii file
        File containing the WCS data for each galaxy
    mosaic_file : str path
        Noise mosaic file that will be created
    imtype : str
        Type of input images to be used; e.g., 'int', 'intbgsub'
    wttype : str
        Type of weight images to be used; e.g., 'rrhr'
    noisetype : str
        Label for noise images; e.g., 'noise'
    window : bool
        Window across input images for pixel-by-pixel noise calculation; Default: False
    """

    # locate the input files from which to calculate noise
    input_noise_dir, imfiles = gather_input_images(gal_dir, this_noise_dir, imtype)

    #now calculate noise
    print('...calculating noise in each input image...')
    for imfile in imfiles:
        if window:
            get_window_val(imfile, input_noise_dir, imtype, noisetype)
        else:
            get_single_val(imfile, input_noise_dir, imtype, noisetype)

    # gather weight images 
    print('...moving weight images...')
    input_noise_wt_dir = gather_weight_images(gal_dir, this_noise_dir, wttype)
    im_dir = input_noise_dir
    wt_dir = input_noise_wt_dir


    # set up directories to house reprojected images
    reproj_noise_dir, reproj_noise_im_dir, reproj_noise_wt_dir = make_dirs(this_noise_dir, imtype, wttype, dirtype='reprojected')

    # reproject the noise and weight images
    print('...reprojecting...')
    reproject_images(gal_hdr.hdrfile_ext, im_dir, reproj_noise_im_dir, noisetype)
    reproject_images(gal_hdr.hdrfile_ext, wt_dir, reproj_noise_wt_dir, '{}_weight'.format(noisetype))
    im_dir = reproj_noise_im_dir
    wt_dir = reproj_noise_wt_dir


    # create metadata tables
    reproj_noise_im_tbl = os.path.join(im_dir, '{}_im_reproj.tbl'.format(noisetype))
    montage.mImgtbl(im_dir, reproj_noise_im_tbl, corners=True)
    reproj_noise_wt_tbl = os.path.join(wt_dir, '{}_wt_reproj.tbl'.format(noisetype))
    montage.mImgtbl(wt_dir, reproj_noise_wt_tbl, corners=True)


    # set up directories to house weighted images
    weight_dir, im_weight_dir, wt_weight_dir = make_dirs(this_noise_dir, imtype, wttype, dirtype='weighted')

    # weight the images
    print('...weighting...')
    im_weight_dir, wt_weight_dir = weight_images(im_dir, wt_dir, weight_dir, imtype=imtype, wttype=wttype, noisetype=noisetype)
    im_dir = im_weight_dir
    wt_dir = wt_weight_dir


    # need to add in quadrature, so copy noise files and square the array
    reproj_square_dir = square_images(this_noise_dir, im_dir, noisetype)
    im_dir = reproj_square_dir


    # now coadd the squared images and also create a count image
    mosaic_noise_dir = os.path.join(this_noise_dir, 'mosaic')
    if not os.path.exists(mosaic_noise_dir):
        os.makedirs(mosaic_noise_dir)

    print('...coadding...')
    coadd(gal_hdr.hdrfile, mosaic_noise_dir, im_dir, reproj_noise_im_tbl, output='{}_im_squared_weighted'.format(noisetype), add_type='mean')
    coadd(gal_hdr.hdrfile, mosaic_noise_dir, im_dir, reproj_noise_im_tbl, output='count', add_type='count')
    coadd(gal_hdr.hdrfile, mosaic_noise_dir, wt_dir, reproj_noise_wt_tbl, output='{}_wt_weighted'.format(noisetype), add_type='mean')


    # multiply the coadded squared image by the numbers in the count image to back out of the 'mean'
    mean_mosaic = os.path.join(mosaic_noise_dir, 'noise_im_squared_weighted_mosaic.fits')
    count_mosaic = os.path.join(mosaic_noise_dir, 'count_mosaic.fits')
    weight_mosaic = os.path.join(mosaic_noise_dir, 'noise_wt_weighted_mosaic.fits')
    mean, mean_hdr = astropy.io.fits.getdata(mean_mosaic, header=True)
    counts, cnt_hdr = astropy.io.fits.getdata(count_mosaic, header=True)
    wt, wt_hdr = astropy.io.fits.getdata(weight_mosaic, header=True)

    total = mean * counts

    # now take the square root of the final image
    final_val = np.sqrt(total) / wt

    # write the final mosaic to file in the mosaic directory
    astropy.io.fits.writeto(mosaic_file, final_val, mean_hdr)


def gather_input_images(gal_dir, this_noise_dir, imtype):
    """
    Gather the input images for noise creation into new input directory within the noise directory
    
    Parameters:
    -----------
    gal_dir : str path
        Path to temporary directory in which mosaic work is being done
    this_noise_dir : str path
        Path to directory within gal_dir where the noise work is completed
    imtype : str
        Type of input images to be used; e.g., 'int', 'intbgsub'

    Returns:
    --------
    input_noise_dir : str
        Path to location of input noise files
    imfiles : str
        List of input files
    """
    im_dir = os.path.join(gal_dir, 'masked', imtype)
    imfiles = sorted(glob.glob(os.path.join(im_dir, '*-{}.fits'.format(imtype))))

    input_noise_dir = os.path.join(this_noise_dir, 'input', imtype)
    if not os.path.exists(input_noise_dir):
        os.makedirs(input_noise_dir)

    return input_noise_dir, imfiles


def get_window_val(imfile, input_noise_dir, imtype, noisetype, outfile=None):
    """
    Get the noise value for each pixel within the desired window.

    Parameters:
    -----------
    imfiles : str
        List of input files for noise calculation
    input_noise_dir : str
        Path to location of input noise files
    imtype : str
        Type of input images to be used; e.g., 'int', 'intbgsub'
    noisetype : str
        Label for noise images; e.g., 'noise'
    outfile : str, optional
        Path to new file that contians noise data for each input file. Will be created if not specified.
    """
    data, hdr = astropy.io.fits.getdata(imfile, header=True)
    sel = np.isnan(data)
    newdata = window(data, size=WINDOW_SIZE)
    newdata[sel] = np.nan
    if outfile is None:
        outfile = os.path.join(input_noise_dir, os.path.basename(imfile).replace(imtype, noisetype))
    astropy.io.fits.writeto(outfile, newdata, hdr)


def window(data, size=WINDOW_SIZE):
    """
    Calculate the noise value for each pixel by taking the standard deviation of all pixels in a window of WINDOW_SIZE.
    
    Parameters:
    -----------
    data : float array
        input data read from fits file
    size : int
        size of window; Default. 30

    Returns:
    --------
    data_std : float array
        The standard deviation of the input data pixel-by-pixel
    """
    def local_std(A):
        return np.nanstd(A)

    data_std = sp.filters.generic_filter(data, local_std, size=size)
    return data_std


def get_single_val(imfile, input_noise_dir, imtype, noisetype, outfile=None):
    """
    Get the single noise value for each input image by taking the standard devation of the image.

    Parameters:
    -----------
    imfiles : str
        List of input files for noise calculation
    input_noise_dir : str
        Path to location of input noise files
    imtype : str
        Type of input images to be used; e.g., 'int', 'intbgsub'
    noisetype : str
        Label for noise images; e.g., 'noise'
    outfile : str, optional
        Path to new file that contians noise data for each input file. Will be created if not specified.
    """
    data, hdr = astropy.io.fits.getdata(imfile, header=True)
    newdata = single_value(data)
    if outfile is None:
        outfile = os.path.join(input_noise_dir, os.path.basename(imfile).replace(imtype, noisetype))
    astropy.io.fits.writeto(outfile, newdata, hdr)


def single_value(data):
    """
    Calculate the noise value for each pixel by taking the standard deviation of all pixels in a window of WINDOW_SIZE.
    
    Parameters:
    -----------
    data : float array
        input data read from fits file

    Returns:
    --------
    newdata : float array
        The single std deviation of the entire image in an array.
    """
    sel = ~np.isnan(data)
    data_std = np.nanstd(data)
    newdata = np.zeros(data.shape)
    newdata[sel] = data_std
    newdata[~sel] = np.nan
    return newdata


def gather_weight_images(gal_dir, this_noise_dir, wttype):
    """
    Gather the weight images used to weight the input files in creation of noise mosaic.

    Parameters:
    -----------
    gal_dir : str path
        Path to temporary directory in which mosaic work is being done
    this_noise_dir : str path
        Path to directory within gal_dir where the noise work is completed
    wttype : str
        Type of weoght images to be used; e.g., 'rrhr'

    Returns:
    --------
    input_noise_wt_dir : str
        Path to location of input noise weight files
    """
    input_noise_wt_dir = os.path.join(this_noise_dir, 'input', 'rrhr')
    if not os.path.exists(input_noise_wt_dir):
        os.makedirs(input_noise_wt_dir)

    wt_dir = os.path.join(gal_dir, 'masked', wttype)
    wtfiles = sorted(glob.glob(os.path.join(wt_dir, '*-{}.fits'.format(wttype))))

    for wtfile in wtfiles:
        if not os.path.exists(os.path.join(input_noise_wt_dir, os.path.basename(wtfile))):
            shutil.copy(wtfile, input_noise_wt_dir)

    return input_noise_wt_dir


def make_dirs(this_noise_dir, imtype, wttype, dirtype='reprojected'):
    """
    Make directories inside noise directory 

    Parameters: 
    -----------
    this_noise_dir : str path
        Path to directory within gal_dir where the noise work is completed
    imtype : str
        Type of input images to be used; e.g., 'int', 'intbgsub'
    wttype : str
        Type of weight images to be used; e.g., 'rrhr'
    dirtype : str
        Name to describe directory

    Returns:
    --------
    noise_dir : str
        Path to newly created directory
    noise_im_dir : str
        Path to im directory within noise_dir
    noise_wt_dir : str
        Path to weight directory within noise_dir
    """
    noise_dir = os.path.join(this_noise_dir, dirtype)
    noise_im_dir = os.path.join(noise_dir, imtype)
    noise_wt_dir = os.path.join(noise_dir, wttype)
    for this_dir in [noise_dir, noise_im_dir, noise_wt_dir]:
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)

    return noise_dir, noise_im_dir, noise_wt_dir


def reproject_images(template_header, input_dir, reproj_dir, imtype, whole=True, exact=True, corners=True, img_list=None):
    """
    Reproject input images to a new WCS as given by a template header

    Parameters
    ----------
    template_header : ascii file
        ASCII file containing the WCS to which you want to reproject. This is what Montage requires.
    input_dir : str
        Path to directory containing input data
    reproj_imtype_dir : 
        Path to new directory for storing reprojected data
    imtype : str
        The type of image you are reprojecting; one of [int, rrhr]
    whole : bool, optional
        Montage argument: Force reprojection of whole images, even if they exceed the area of the FITS 
        header template (Default: True)
    exact : bool, optional
        Montage argument: Flag indicating output image should exactly match the FITS header template, 
        and not crop off blank pixels (Default: True)
    corners : bool, optional
        Montage argument: Adds 8 columns for the RA and Dec of the image corners to the output metadata table 
        (Default: True)
    img_list : list of strs, optional 
        Montage argument: only process files with names specified in table img_list, ignoring any other files
        in the directory. (Default: None)
    """

    # get image metadata from input images
    input_table = os.path.join(input_dir, imtype + '_input.tbl')
    montage.mImgtbl(input_dir, input_table, corners=corners, img_list=img_list)

    # reproject images
    stats_table = os.path.join(reproj_dir, imtype+'_mProjExec_stats.log')
    montage.mProjExec(input_table, template_header, reproj_dir, stats_table, raw_dir=input_dir, 
                      whole=whole, exact=exact)

    # get new image metadata with new header information
    reprojected_table = os.path.join(reproj_dir, imtype + '_reprojected.tbl')
    montage.mImgtbl(reproj_dir, reprojected_table, corners=corners)


def weight_images(im_dir, wt_dir, weight_dir, imtype='-int', wttype='-rrhr', noisetype='noise'):
    """
    Weight the input images by a set of weights images

    Parameters
    ----------
    im_dir : str
        Path to directory containing the images
    wt_dir : str
        Path to directory containing the weights
    weight_dir : str
        Path to directory for the newly weighted images
    im_weight_dir : str
        Path to subdirectory containing the weighted images
    wt_weight_dir : str
        Path to subdirectory containgn the weights images (same as before, they haven't changed)    
    """
    im_suff, wt_suff = '*-{}.fits'.format(noisetype), '*-{}.fits'.format(wttype)
    imfiles = sorted(glob.glob(os.path.join(im_dir, im_suff)))
    wtfiles = sorted(glob.glob(os.path.join(wt_dir, wt_suff)))

    im_weight_dir = os.path.join(weight_dir, imtype)
    wt_weight_dir = os.path.join(weight_dir, wttype)
    for thisdir in [im_weight_dir, wt_weight_dir]:
        if not os.path.exists(thisdir):
            os.makedirs(thisdir)

    # weight each image
    for i in range(len(imfiles)):
        # read in the data
        imfile = imfiles[i]
        wtfile = os.path.join(os.path.dirname(wtfiles[i]), os.path.basename(imfile).replace(noisetype, wttype))
        im, hdr = astropy.io.fits.getdata(imfile, header=True)
        rrhr, rrhrhdr = astropy.io.fits.getdata(wtfile, header=True)

        # weight the data by the exposure time
        wt = rrhr
        newim = im * wt

        # write data to new files and copy the *_area.fits files created by Montage to have the same naming convention
        newfile = os.path.join(im_weight_dir, os.path.basename(imfile))
        astropy.io.fits.writeto(newfile, newim, hdr)
        old_area_file = imfile.replace('.fits', '_area.fits')
        if os.path.exists(old_area_file):
            new_area_file = newfile.replace('.fits', '_area.fits')
            shutil.copy(old_area_file, new_area_file)

        weightfile = os.path.join(wt_weight_dir, os.path.basename(wtfile))
        astropy.io.fits.writeto(weightfile, wt, rrhrhdr)
        old_area_file = wtfile.replace('.fits', '_area.fits')
        if os.path.exists(old_area_file):
            new_area_file = weightfile.replace('.fits', '_area.fits')
            shutil.copy(old_area_file, new_area_file)

    return im_weight_dir, wt_weight_dir


def square_images(this_noise_dir, im_dir, noisetype):
    """
    Square the reprojected, weighted images

    Parameters:
    -----------
    this_noise_dir : str
        Path to directory within gal_dir where the noise work is completed
    im_dir : str
        Path to directory containing the reprojected, weighted images
    noisetype : str
        Label for noise images; e.g., 'noise'

    Returns:
    --------
    reproj_square_dir : str
        Path to directory containing the squared images

    """
    reproj_square_dir = os.path.join(this_noise_dir, 'reprojected_weighted_squared')
    if not os.path.exists(reproj_square_dir):
        os.makedirs(reproj_square_dir)

    copyfiles = glob.glob(os.path.join(im_dir, '*-{}.fits'.format(noisetype)))
    for copyfile in copyfiles:
        shutil.copy(copyfile, reproj_square_dir)

    # square the data
    print('...squaring...')
    imfiles = glob.glob(os.path.join(reproj_square_dir, '*-{}.fits'.format(noisetype)))
    for imfile in imfiles:
        data, hdr = astropy.io.fits.getdata(imfile, header=True)
        newdata = data ** 2
        try:
            astropy.io.fits.writeto(imfile, newdata, hdr, overwrite=True)
        except:
            astropy.io.fits.writeto(imfile, newdata, hdr, clobber=True)

    return reproj_square_dir


def coadd(template_header, output_dir, input_dir, reprojected_table, output='noise', add_type=None):
    """
    Coadd input images to create mosaic.

    Parameters
    ----------
    template_header : ascii file
        File containing new WCS
    output_dir : str
        Path to directory contianing the output image
    input_dir : str
        path to directory containing the input images
    output : str, optional
        Type of mosaic you're making: e.g., int, wt, count (Default: None)
    add_type : str, optional
        Montage argument -- type of coadding to perform (Default: None -- defaults to Montage's default)
    """
    #reprojected_table = os.path.join(input_dir, output_type + '_reprojected.tbl')
    out_image = os.path.join(output_dir, output + '_mosaic.fits')
    montage.mAdd(reprojected_table, template_header, out_image, img_dir=input_dir, exact=True, type=add_type)





def get_args():
    """ Get command line arguments """
    parser = argparse.ArgumentParser(description='Create cutouts of a given size around each galaxy center.')
    parser.add_argument('--imtype', default='intbgsub', help='input images to use. Default: intbgsub')
    parser.add_argument('--wttype', default='rrhr', help='images to use for the weighting. Default: rrhr')
    parser.add_argument('--size', default=30, help='cutout size in arcminutes. Default: 30.')
    parser.add_argument('--desired_pix_scale', default=1.5, help='desired pixel scale of output image. Default: 1.5 (GALEX)')
    parser.add_argument('--band', default=None, help='waveband. Default: None (does all)')
    parser.add_argument('--model_bg', default=True, help='model the background to match all images as best as possible. Default: True')
    parser.add_argument('--weight_ims', default=True, help='weight the input images by the desired weight images. Default: True')
    parser.add_argument('--convert_mjysr', default=True, help='convert images to MJy/sr. Default: True')
    parser.add_argument('--galaxy_list', default=None, nargs='+', help='Galaxy name if doing a single cutout or list of names. Default: None')
    parser.add_argument('--all_galaxies', action='store_true', help='run all galaxies in database. Default: False; include flag to store_true')
    parser.add_argument('--tag', default=None, help='tag to select galaxies, i.e., SINGS, HERACLES, etc. Default: None')
    parser.add_argument('--inds', nargs=2, type=int, help='index the all galaxy array')
    parser.add_argument('--window', action='store_true', help='window across input images when creating noise mosaic. Default: False')
    return parser.parse_args()


def main(**kwargs):
    """ Create noise mosaics for the GALEX cutouts
    
    Parameters
    ----------
    imtype : str
        input image type to use from galex (Default: intbgsub)
    wttype : str
        input weights image type to use from galex (Default: rrhr)
    size : float
        cutout size in arcminutes (Default: 30.0)
    desired_pix_scale : float
        Desired pixel scale of output image. Default is currently set to GALEX pixel scale (Default: 1.5)
    band : str
        the band in which the cutout is made, either a single band or a list (Default: fuv)
    model_bg : bool
        model the background in Montage (Default: True)
    weight_ims : bool
        weight the input images with the weights images (Default: True)
    convert_mjysr : bool
        convert input images from counts/sec to MJy/sr (Default: True)
    galaxy_list : list
        list of one or more galaxies for which to make cutouts. Do not set if you want to make cutouts for all galaxies (Default: None)
    all_galaxies : bool
        Make cutouts for all galaxies in the galbase (Default: False)
    tag : str
        A tag to select a subset of galaxies; i.e., SINGS, HERACLES, etc. (Default: None)
    inds : int
        List of two ints to index the galaxy array from [int1:int2]
    window : bool
        Use a window across each input image to calculate pixel-by-pixel noise instead of single value for entire image

    Example:
    This code can be run from the command line or imported and run within a separate program.
    The following example creates noise mosaics of the SINGS sample that are 30x30 arcminutes 
    in the FUV using a single standard deviation for each input image rather than pixel-by-pixel.

    Usage:
    %run noise_mosaics.py --band fuv --tag SINGS

    or

    import noise_mosaics
    noise_mosaics.main(size=30, band='fuv', tag='SINGS')

    To get pixel-by-pixel noise, use the --window flag in the first call or window=True in the second.
    """

    warnings.filterwarnings('ignore')
    wband = kwargs['band']
    if kwargs['inds']:
        kwargs['all_galaxies'] = True

    #get data from galbase
    gals = gal_data.gal_data(names=kwargs['galaxy_list'], data=None, all=kwargs['all_galaxies'], 
                             galdata_dir=_GALDATA_DIR, tag=kwargs['tag']) 

    if kwargs['inds']:
        ind_start, ind_stop = kwargs['inds'][0], kwargs['inds'][1]
        gals = gals[ind_start:ind_stop]

    n_gals = len(gals)
    size_deg = kwargs['size'] * 60. / 3600. #convert from arcminutes to degrees

    for i in range(n_gals):
        galname = gals['name'][i].replace(' ', '').upper()
        pgcname = gals['pgcname'][i]
        ra_ctr, dec_ctr = gals['ra_deg'][i], gals['dec_deg'][i]

        stamp_kwargs = {'ra_ctr': ra_ctr, 'dec_ctr': dec_ctr, 'size_deg': size_deg, 'name': galname, 
                        'pgcname': pgcname, 'model_bg': kwargs['model_bg'], 
                        'weight_ims': kwargs['weight_ims'], 'convert_mjysr': kwargs['convert_mjysr'], 
                        'imtype': kwargs['imtype'], 'wttype': kwargs['wttype'], 
                        'desired_pix_scale': kwargs['desired_pix_scale'], 'window': kwargs['window']}
        if wband == 'fuv':
            make_mosaic(band='fuv', **stamp_kwargs)
        elif wband == 'nuv':
            make_mosaic(band='nuv', **stamp_kwargs)
        else:
            make_mosaic(band='fuv', **stamp_kwargs)
            make_mosaic(band='nuv', **stamp_kwargs)



if __name__ == '__main__':
    args = get_args()
    main(**vars(args))





