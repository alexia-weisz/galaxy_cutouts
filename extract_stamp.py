import astropy.io.fits
import astropy.wcs
import os
import sys
import numpy as np
import montage_wrapper as montage
import shutil
import sys
import glob
import time
from matplotlib.path import Path
from pdb import set_trace


_TOP_DIR = '/data/tycho/0/leroy.42/allsky/'
_INDEX_DIR = os.path.join(_TOP_DIR, 'z0mgs/')
_HOME_DIR = '/data/tycho/0/lewis.1590/atlas/'
_MOSAIC_DIR = os.path.join(_HOME_DIR, 'cutouts')


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


def make_axes(hdr, quiet=False, novec=False, vonly=False, simple=False):
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


def unwise(band=None, ra_ctr=None, dec_ctr=None, size_deg=None, index=None, name=None):
    tel = 'unwise'
    data_dir = os.path.join(_TOP_DIR, tel, 'sorted_tiles')

    # READ THE INDEX FILE (IF NOT PASSED IN)
    if index is None:
        indexfile = os.path.join(_INDEX_DIR, tel + '_index_file.fits')
        ext = 1
        index, hdr = astropy.io.fits.getdata(indexfile, ext, header=True)

    # CALIBRATION TO GO FROM VEGAS TO ABMAG
    w1_vtoab = 2.683
    w2_vtoab = 3.319
    w3_vtoab = 5.242
    w4_vtoab = 6.604

    # NORMALIZATION OF UNITY IN VEGA MAG
    norm_mag = 22.5
    pix_as = 2.75  #arcseconds - native detector pixel size wise docs

    # COUNTS TO JY CONVERSION
    w1_to_mjysr = counts2jy(norm_mag, w1_vtoab, pix_as)
    w2_to_mjysr = counts2jy(norm_mag, w2_vtoab, pix_as)
    w3_to_mjysr = counts2jy(norm_mag, w3_vtoab, pix_as)
    w4_to_mjysr = counts2jy(norm_mag, w4_vtoab, pix_as)

    # MAKE A HEADER
    pix_scale = 2.0 / 3600.  # 2.0 arbitrary
    pix_len = size_deg / pix_scale

    # this should automatically populate SIMPLE and NAXIS keywords
    target_hdr = create_hdr(ra_ctr, dec_ctr, pix_len, pix_scale)

    # CALCULATE TILE OVERLAP
    tile_overlaps = calc_tile_overlap(ra_ctr, dec_ctr, pad=size_deg,
                                      min_ra=index['MIN_RA'],
                                      max_ra=index['MAX_RA'],
                                      min_dec=index['MIN_DEC'],
                                      max_dec=index['MAX_DEC'])

    # FIND OVERLAPPING TILES WITH RIGHT BAND
    #  index file set up such that index['BAND'] = 1, 2, 3, 4 depending on wise band
    ind = np.where((index['BAND'] == band) & tile_overlaps)
    ct_overlap = len(ind[0])

    # SET UP THE OUTPUT
    ri_targ, di_targ = make_axes(target_hdr)
    sz_out = ri_targ.shape
    outim = ri_targ * np.nan

    # LOOP OVER OVERLAPPING TILES AND STITCH ONTO TARGET HEADER
    for ii in range(0, ct_overlap):
        infile = os.path.join(data_dir, index[ind[ii]]['FNAME'])
        im, hdr = astropy.io.fits.getdata(infile, header=True)
        ri, di = make_axes(hdr)

        hh = astropy.wcs.WCS(target_hdr)
        x, y = ww.all_world2pix(zip(ri, di), 1)

        in_image = (x > 0 & x < (sz_out[0]-1)) & (y > 0 and y < (sz_out[1]-1))
        if np.sum(in_image) == 0:
            print("No overlap. Proceeding.")
            continue

        if band == 1:
            im *= w1_to_mjysr
        if band == 2:
            im *= w2_to_mjysr
        if band == 3:
            im *= w3_to_mjysr
        if band == 4:
            im *= w4_to_mjysr

        target_hdr['BUNIT'] = 'MJY/SR'

        newimfile = reprojection(infile, im, hdr, target_hdr, data_dir)
        im, new_hdr = astropy.io.fits.getdata(newimfile, header=True)

        useful = np.where(np.isfinite(im))
        outim[useful] = im[useful]

        return outim, target_hdr


def counts2jy(norm_mag, calibration_value, pix_as):
    """
    Convert counts to Jy -- this is from Adam's unwise stuff

    Parameters
    ----------
    norm_mag : float
        input data
    calibration_value : float
        Value for converting from counts to mag
    pix_as : float
        Pixel scale in arcseconds

    Returns
    -------
    val : float
        Converted data, now in MJy/sr
    """
    # convert counts to Jy
    val = 10.**((norm_mag + calibration_value) / -2.5)
    val *= 3631.0
    # then to MJy
    val /= 1e6
    # then to MJy/sr
    val /= np.radians(pix_as / 3600.)**2
    return val


def write_headerfile(header_file, header):
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


def create_hdr(ra_ctr, dec_ctr, pix_len, pix_scale):
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
    hdr['CRVAL1'] = float(ra_ctr)
    hdr['CRPIX1'] = (pix_len / 2.) * 1.
    hdr['CDELT1'] = -1.0 * pix_scale
    hdr['CTYPE2'] = 'DEC--TAN'
    hdr['CRVAL2'] = float(dec_ctr)
    hdr['CRPIX2'] = (pix_len / 2.) * 1.
    hdr['CDELT2'] = pix_scale
    hdr['EQUINOX'] = 2000
    return hdr


def galex(band='fuv', ra_ctr=None, dec_ctr=None, size_deg=None, index=None, name=None, write_info=True, model_bg=False):
    """
    Create cutouts of a galaxy in a single GALEX band.

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
    write_info : bool, optional
        Write info about each mosaic to file (Default: True) -- NO LONGER USED?
    model_bg : bool, optional
        Model the background of the mosaiced image (Default: False)
    """
    ttype = 'galex'
    data_dir = os.path.join(_TOP_DIR, ttype, 'sorted_tiles')
    problem_file = os.path.join(_HOME_DIR, 'problem_galaxies_' + band + '.txt')
    bg_reg_file = os.path.join(_HOME_DIR, 'galex_reprojected_bg.reg')
    numbers_file = os.path.join(_HOME_DIR, 'gal_reproj_info_' + band + '.dat')

    galaxy_mosaic_file = os.path.join(_MOSAIC_DIR, '_'.join([name, band]).upper() + '.FITS')

    start_time = time.time()
    if not os.path.exists(galaxy_mosaic_file):
        print name, band.upper()

        # READ THE INDEX FILE (IF NOT PASSED IN)
        if index is None:
            indexfile = os.path.join(_INDEX_DIR, ttype + '_index_file.fits')
            ext = 1
            index, hdr = astropy.io.fits.getdata(indexfile, ext, header=True)

        # CALIBRATION FROM COUNTS TO ABMAG
        fuv_toab = 18.82
        nuv_toab = 20.08

        # PIXEL SCALE IN ARCSECONDS
        pix_as = 1.5  # galex pixel scale -- from galex docs

        # MAKE A HEADER
        pix_scale = 1.5 / 3600.  # 1.5 arbitrary: how should I set it?
        pix_len = size_deg / pix_scale
        target_hdr = create_hdr(ra_ctr, dec_ctr, pix_len, pix_scale)


        # MAKE EXTENDED HEADER
        pix_len_ext = int(np.ceil(size_deg * 3 / pix_scale))
        target_hdr_ext = create_hdr(ra_ctr, dec_ctr, pix_len_ext, pix_scale)


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
                myfile.write(name + ': ' + 'No overlapping tiles\n')
            return

        # SET UP THE OUTPUT
        ri_targ, di_targ = make_axes(target_hdr)
        sz_out = ri_targ.shape
        outim = ri_targ * np.nan
        prihdu = astropy.io.fits.PrimaryHDU(data=outim, header=target_hdr)
        target_hdr = prihdu.header
        
        ri_targ_ext, di_targ_ext = make_axes(target_hdr_ext)
        sz_out = ri_targ_ext.shape
        outim = ri_targ_ext * np.nan
        prihdu_ext = astropy.io.fits.PrimaryHDU(data=outim, header=target_hdr_ext)
        target_hdr_ext = prihdu_ext.header

        try:
            # CREATE NEW TEMP DIRECTORY TO STORE TEMPORARY FILES
            gal_dir = os.path.join(_HOME_DIR, '_'.join([name, band]).upper())
            os.makedirs(gal_dir)


            # GATHER THE INPUT FILES
            im_dir, wt_dir, nfiles = get_input(index, ind, data_dir, gal_dir)


            # CONVERT INT FILES TO MJY/SR AND WRITE NEW FILES INTO TEMP DIR
            im_dir, wt_dir = convert_files(gal_dir, im_dir, wt_dir, band, fuv_toab, nuv_toab, pix_as)


            # APPEND UNIT INFORMATION TO THE NEW HEADER AND WRITE OUT HEADER FILE
            hdrs = [target_hdr, target_hdr_ext]
            suff = ['_template.hdr','_template_ext.hdr']
            basef = [name + hh for hh in suff]
            hdr_files = [os.path.join(gal_dir, b) for b in  basef]
            for i, tg in enumerate(hdrs):
                tg['BUNIT'] = 'MJY/SR'
                write_headerfile(hdr_files[i], tg)


            # MASK IMAGES
            im_dir, wt_dir = mask_images(im_dir, wt_dir, gal_dir)


            # REPROJECT IMAGES
            reprojected_dir = os.path.join(gal_dir, 'reprojected')
            os.makedirs(reprojected_dir)
            im_dir = reproject_images(hdr_files[1], im_dir, reprojected_dir, 'int')
            wt_dir = reproject_images(hdr_files[1], wt_dir, reprojected_dir,'rrhr')


            # MODEL THE BACKGROUND IN THE IMAGE FILES?
            if model_bg:
                im_dir = bg_model(gal_dir, im_dir, hdr_files[1], level_only=True)


            # WEIGHT IMAGES
            weight_dir = os.path.join(gal_dir, 'weight')
            os.makedirs(weight_dir)
            im_dir, wt_dir = weight_images(im_dir, wt_dir, weight_dir)


            # CREATE THE METADATA TABLES NEEDED FOR COADDITION
            weight_table = create_table(wt_dir, dir_type='weights')
            weighted_table = create_table(im_dir, dir_type='int')
            count_table = create_table(im_dir, dir_type='count')


            # COADD THE REPROJECTED, WEIGHTED IMAGES AND THE WEIGHT IMAGES
            final_dir = os.path.join(gal_dir, 'mosaic')
            os.makedirs(final_dir)
            coadd(hdr_files[0], final_dir, wt_dir, output='weights')
            coadd(hdr_files[0], final_dir, im_dir, output='int')
            coadd(hdr_files[0], final_dir, im_dir, output='count', add_type='count')


            # DIVIDE OUT THE WEIGHTS
            imagefile = finish_weight(final_dir)


            # SUBTRACT OUT THE BACKGROUND
            rm_overall_bg = False
            if rm_overall_bg:
                remove_background(final_dir, imagefile, bg_reg_file)
            else:
                outfile = os.path.join(final_dir, 'final_mosaic.fits')
                shutil.copy(imagefile, outfile)


            # COPY MOSAIC FILES TO CUTOUTS DIRECTORY
            mosaic_file = os.path.join(final_dir, 'final_mosaic.fits')
            weight_file = os.path.join(final_dir, 'weights_mosaic.fits')
            count_file = os.path.join(final_dir, 'count_mosaic.fits')
            newfile = '_'.join([name, band]).upper() + '.FITS'
            wt_file = '_'.join([name, band]).upper() + '_weight.FITS'
            ct_file = '_'.join([name, band]).upper() + '_count.FITS'
            new_mosaic_file = os.path.join(_MOSAIC_DIR, newfile)
            new_weight_file = os.path.join(_MOSAIC_DIR, wt_file)
            new_count_file = os.path.join(_MOSAIC_DIR, ct_file)
            shutil.copy(mosaic_file, new_mosaic_file)
            shutil.copy(weight_file, new_weight_file)
            shutil.copy(count_file, new_count_file)


            # REMOVE TEMP GALAXY DIRECTORY AND EXTRA FILES
            shutil.rmtree(gal_dir, ignore_errors=True)


            # NOTE TIME TO FINISH
            stop_time = time.time()
            total_time = (stop_time - start_time) / 60.


            # WRITE OUT THE NUMBER OF TILES THAT OVERLAP THE GIVEN GALAXY
            out_arr = [name, band.upper(), nfiles, np.around(total_time, 2)]
            with open(numbers_file, 'a') as nfile:
                nfile.write('{0: >10}'.format(out_arr[0]))
                nfile.write('{0: >6}'.format(out_arr[1]))
                nfile.write('{0: >6}'.format(out_arr[2]))
                nfile.write('{0: >6}'.format(out_arr[3]) + '\n')
                #nfile.write(name + ': ' + str(len(infiles)) + '\n')

        # SOMETHING WENT WRONG -- WRITE ERROR TO FILE
        except Exception as inst:
            me = sys.exc_info()[0]
            with open(problem_file, 'a') as myfile:
                myfile.write(name + ': ' + str(me) + ': '+str(inst)+'\n')
            shutil.rmtree(gal_dir, ignore_errors=True)

    return


def get_input(index, ind, data_dir, gal_dir):
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
    gal_dir : str
        Path to newly created temporary directory for storing temp files used in mosaicing

    Returns
    -------
    input_dir : str
        The path to the input files
    len(input_files) : int
        The number of files that will go into the mosaic.
    """
    input_dir = os.path.join(gal_dir, 'input')
    os.makedirs(input_dir)
    infiles = index[ind[0]]['fname']
    wtfiles = index[ind[0]]['rrhrfile']
    flgfiles = index[ind[0]]['flagfile']
    infiles = [os.path.join(data_dir, f) for f in infiles]
    wtfiles = [os.path.join(data_dir, f) for f in wtfiles]
    flgfiles = [os.path.join(data_dir, f) for f in flgfiles]

    for infile in infiles:
        basename = os.path.basename(infile)
        new_in_file = os.path.join(input_dir, basename)
        os.symlink(infile, new_in_file)

    for wtfile in wtfiles:
        basename = os.path.basename(wtfile)
        new_wt_file = os.path.join(input_dir, basename)
        os.symlink(wtfile, new_wt_file)

    for flgfile in flgfiles:
        basename = os.path.basename(flgfile)
        new_flg_file = os.path.join(input_dir, basename)
        os.symlink(flgfile, new_flg_file)

    return input_dir, input_dir, len(infiles)


def convert_files(gal_dir, im_dir, wt_dir, band, fuv_toab, nuv_toab, pix_as):
    """
    Convert GALEX files from cts/sec to MJy/sr

    Parameters
    ----------
    gal_dir : str
        Path to temp directory in which mosaicing is being performed
    im_dir : str
        Path to directory that holds the input images (-int files in GALEX)
    wt_dir : str
        Path to directory that holds the input weights images (-rrhr files in GALEX)
    band : str
        waveband to use: fuv or nuv
    fuv_toab : float
        GALEX FUV conversion from counts to AB mag
    nuv_toab : float
        GALEX NUV conversion from counts to AB mag
    pix_as : float
        pixel scale in arcseconds

    Returns
    -------
    converted_dir : str
        Path to directory containing images converted to flux density
    """
    converted_dir = os.path.join(gal_dir, 'converted')
    os.makedirs(converted_dir)

    intfiles = sorted(glob.glob(os.path.join(im_dir, '*-int.fits')))
    wtfiles = sorted(glob.glob(os.path.join(wt_dir, '*-rrhr.fits')))

    int_outfiles = [os.path.join(converted_dir, os.path.basename(f).replace('.fits', '_mjysr.fits')) for f in intfiles]
    wt_outfiles = [os.path.join(converted_dir, os.path.basename(f)) for f in wtfiles]

    for i in range(len(intfiles)):
        if os.path.exists(wtfiles[i]):
            im, hdr = astropy.io.fits.getdata(intfiles[i], header=True)
            wt, whdr = astropy.io.fits.getdata(wtfiles[i], header=True)

            if band.lower() == 'fuv':
                im = counts2jy_galex(im, fuv_toab, pix_as)
            if band.lower() == 'nuv':
                im = counts2jy_galex(im, nuv_toab, pix_as)
            if not os.path.exists(int_outfiles[i]):
                #im -= np.mean(im)  # subtract out the mean of each image
                astropy.io.fits.writeto(int_outfiles[i], im, hdr)
            if not os.path.exists(wt_outfiles[i]):
                astropy.io.fits.writeto(wt_outfiles[i], wt, whdr)
        else:
            continue

    return converted_dir, converted_dir


def mask_images(im_dir, wt_dir, gal_dir):
    """
    Mask pixels in the input images

    Parameters
    ----------
    im_dir : str
        Path to directory containing the images
    wt_dir : str
        Path to directory containing the weights
    gal_dir : str
        Path to temp directory for this galaxy in which the mosaicing is being performed

    Returns
    -------
    int_masked_dir : str
        Path to directory containing the masked images
    wt_masked_dir : str
        Path to directory containing the masked weight images
    """
    masked_dir = os.path.join(gal_dir, 'masked')
    os.makedirs(masked_dir)

    int_masked_dir = os.path.join(masked_dir, 'int')
    wt_masked_dir = os.path.join(masked_dir, 'rrhr')
    os.makedirs(int_masked_dir)
    os.makedirs(wt_masked_dir)

    int_suff, rrhr_suff = '*_mjysr.fits', '*-rrhr.fits'
    int_images = sorted(glob.glob(os.path.join(im_dir, int_suff)))
    rrhr_images = sorted(glob.glob(os.path.join(wt_dir, rrhr_suff)))

    for i in range(len(int_images)):
        image_infile = int_images[i]
        wt_infile = rrhr_images[i]

        image_outfile = os.path.join(int_masked_dir, os.path.basename(image_infile))
        wt_outfile = os.path.join(wt_masked_dir, os.path.basename(wt_infile))

        mask_galex(image_infile, wt_infile, out_intfile=image_outfile, out_wtfile=wt_outfile)

    return int_masked_dir, wt_masked_dir


def mask_galex(intfile, wtfile, chip_rad=1400, chip_x0=1920, chip_y0=1920, out_intfile=None, out_wtfile=None):
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
    if out_intfile is None:
        out_intfile = intfile.replace('.fits', '_masked.fits')
    if out_wtfile is None:
        out_wtfile = wtfile.replace('.fits', '_masked.fits')

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


def reproject_images(template_header, input_dir, reprojected_dir, imtype, whole=True, exact=True, img_list=None):
    """
    Reproject input images to a new WCS as given by a template header

    Parameters
    ----------
    template_header : ascii file
        ASCII file containing the WCS to which you want to reproject. This is what Montage requires.
    input_dir : str
        Path to directory containing input data
    reprojected_dir : 
        Path to new directory for storing reprojected data
    imtype : str
        The type of image you are reprojecting; one of [int, rrhr]
    whole : bool, optional
        Montage argument (Default: True)
    exact : bool, optional
        Montage argument (Default: True)
    img_list : list of strs, optional 
        Montage argument (Default: None)

    Returns
    -------
    reproj_imtype_dir : str
        Path to output directory containing the reprojected images
    """
    # create directory for storing reprojected images
    reproj_imtype_dir = os.path.join(reprojected_dir, imtype)
    os.makedirs(reproj_imtype_dir)

    # get image metadata from input images
    input_table = os.path.join(input_dir, imtype + '_input.tbl')
    montage.mImgtbl(input_dir, input_table, corners=True, img_list=img_list)

    # Create reprojection directory, reproject, and get image metadata
    stats_table = os.path.join(reproj_imtype_dir, imtype+'_mProjExec_stats.log')
    montage.mProjExec(input_table, template_header, reproj_imtype_dir, stats_table, raw_dir=input_dir, whole=whole, exact=exact)
    reprojected_table = os.path.join(reproj_imtype_dir, imtype + '_reprojected.tbl')
    montage.mImgtbl(reproj_imtype_dir, reprojected_table, corners=True)

    return reproj_imtype_dir


def bg_model(gal_dir, reprojected_dir, template_header, level_only=False):
    """
    Model the background for the mosaiced image

    Parameters
    ----------
    gal_dir : str
        Path to temp directory containing all data for galaxy 
    reprojected_dir : str
        Path to directory inside gal_dir containing the reprojected images
    template_header : ascii file
        Path to file containing the WCS to which we want to reproject our images
    level_only : bool, optional
        Montage argument: Adjust background levels only, don't try to fit the slope (Default: False)

    Returns
    -------
    corr_dir : str
        Path to directory containing the background-corrected images
    """
    bg_model_dir = os.path.join(gal_dir, 'background_model')
    os.makedirs(bg_model_dir)

    # FIND OVERLAPS
    diff_dir = os.path.join(bg_model_dir, 'differences')
    os.makedirs(diff_dir)
    reprojected_table = os.path.join(reprojected_dir,'int_reprojected.tbl')
    diffs_table = os.path.join(diff_dir, 'differences.tbl')
    montage.mOverlaps(reprojected_table, diffs_table)


    # CALCULATE DIFFERENCES BETWEEN OVERLAPPING IMAGES
    montage.mDiffExec(diffs_table, template_header, diff_dir,
                      proj_dir=reprojected_dir)


    # BEST-FIT PLANE COEFFICIENTS
    fits_table = os.path.join(diff_dir, 'fits.tbl')
    montage.mFitExec(diffs_table, fits_table, diff_dir)


    # CALCULATE CORRECTIONS
    corr_dir = os.path.join(bg_model_dir, 'corrected')
    os.makedirs(corr_dir)
    corrections_table = os.path.join(corr_dir, 'corrections.tbl')
    montage.mBgModel(reprojected_table, fits_table, corrections_table,
                     level_only=level_only)


    # APPLY CORRECTIONS
    montage.mBgExec(reprojected_table, corrections_table, corr_dir,
                    proj_dir=reprojected_dir)

    return corr_dir


def weight_images(im_dir, wt_dir, weight_dir):
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

    Returns
    -------
    im_weight_dir : str
        Path to subdirectory containing the weighted images
    wt_weight_dir : str
        Path to subdirectory containgn the weights images (same as before, they haven't changed)
    """
    im_suff, wt_suff = '*_mjysr.fits', '*-rrhr.fits'
    imfiles = sorted(glob.glob(os.path.join(im_dir, im_suff)))
    wtfiles = sorted(glob.glob(os.path.join(wt_dir, wt_suff)))

    # create the new output directories
    im_weight_dir = os.path.join(weight_dir, 'int')
    wt_weight_dir = os.path.join(weight_dir, 'rrhr')
    [os.makedirs(out_dir) for out_dir in [im_weight_dir, wt_weight_dir]]

    # weight each image
    for i in range(len(imfiles)):
        # read in the data
        imfile = imfiles[i]
        wtfile = wtfiles[i]
        im, hdr = astropy.io.fits.getdata(imfile, header=True)
        rrhr, rrhrhdr = astropy.io.fits.getdata(wtfile, header=True)

        # weight the data by the exposure time
        # noise = 1. / np.sqrt(rrhr)
        # weight = 1 / noise**2
        wt = rrhr
        newim = im * wt

        # write data to new files and copy the *_area.fits files created by Montage to have the same naming convention
        #nf = imfiles[i].split('/')[-1].replace('.fits', '_weighted.fits')
        #newfile = os.path.join(weighted_dir, nf)
        newfile = os.path.join(im_weight_dir, os.path.basename(imfile))
        astropy.io.fits.writeto(newfile, newim, hdr)
        old_area_file = imfile.replace('.fits', '_area.fits')
        if os.path.exists(old_area_file):
            new_area_file = newfile.replace('.fits', '_area.fits')
            shutil.copy(old_area_file, new_area_file)

        #nf = wtfiles[i].split('/')[-1].replace('.fits', '_weights.fits')
        #weightfile = os.path.join(weights_dir, nf)
        weightfile = os.path.join(wt_weight_dir, os.path.basename(wtfile))
        astropy.io.fits.writeto(weightfile, wt, rrhrhdr)
        old_area_file = wtfile.replace('.fits', '_area.fits')
        if os.path.exists(old_area_file):
            new_area_file = weightfile.replace('.fits', '_area.fits')
            shutil.copy(old_area_file, new_area_file)

    return im_weight_dir, wt_weight_dir


def create_table(in_dir, dir_type=None):
    """
    Create a metadata table using Montage for all the files in a given directory

    Parameters
    ----------
    in_dir : str
        Path to directory containing the files
    dir_type : str, optional
        type of file you are creating a table for, e.g., 'int, rrhr, wt' (Default: None)

    Returns
    -------
    reprojected_table 
        Path to the table containing the metadata
    """
    if dir_type is None:
        reprojected_table = os.path.join(in_dir, 'reprojected.tbl')
    else:
        reprojected_table = os.path.join(in_dir, dir_type + '_reprojected.tbl')
    montage.mImgtbl(in_dir, reprojected_table, corners=True)
    return reprojected_table


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
        Converted count rate data
    """
    # first convert to abmag
    abmag = -2.5 * np.log10(counts) + cal

    # then convert to Jy
    f_nu = 10**(abmag/-2.5) * 3631.

    # then to MJy
    f_nu *= 1e-6

    # then to MJy/sr
    val = f_nu / (np.radians(pix_as/3600.))**2
    return val
    #val = flux / MJYSR2JYARCSEC / pixel_area / 1e-23 / C * FUV_LAMBDA**2


def wtpersr(wt, pix_as):
    """
    Convert weights data to per steradian. NOT USED
    """
    return wt / (np.radians(pix_as/3600.))**2


def coadd(template_header, output_dir, input_dir, output=None, add_type=None):
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
    img_dir = input_dir
    # output is either 'weights' or 'int'
    if output is None:
        reprojected_table = os.path.join(img_dir, 'reprojected.tbl')
        out_image = os.path.join(output_dir, 'mosaic.fits')
    else:
        reprojected_table = os.path.join(img_dir, output + '_reprojected.tbl')
        out_image = os.path.join(output_dir, output + '_mosaic.fits')
    montage.mAdd(reprojected_table, template_header, out_image, img_dir=img_dir, exact=True, type=add_type)


def finish_weight(output_dir):
    """
    Divide out the weights from the final image to get back to flux density units

    Parameters
    ----------
    output_dir : str
        Path to directory containing the output image

    Returns
    -------
    newfile : str
        Path to new, unweighted mosaiced file
    """
    image_file = os.path.join(output_dir, 'int_mosaic.fits')
    wt_file = os.path.join(output_dir, 'weights_mosaic.fits')
    count_file = os.path.join(output_dir, 'count_mosaic.fits')
    im, hdr = astropy.io.fits.getdata(image_file, header=True)
    wt = astropy.io.fits.getdata(wt_file)
    ct = astropy.io.fits.getdata(count_file)

    newim = im / wt

    newfile = os.path.join(output_dir, 'image_mosaic.fits')
    astropy.io.fits.writeto(newfile, newim, hdr)

    return newfile


def remove_background(final_dir, imfile, bgfile):
    """
    Remove a background from the mosaiced image

    Parameters
    ----------
    final_dir : str
        Path to directory that will contain the final image
    imfile : str
        Path to file from which you want to remove a background
    bgfile : str 
        Path to file from which to calculate a background
    """
    # read in the data
    data, hdr = astropy.io.fits.getdata(imfile, header=True)
    box_inds = read_bg_regfile(bgfile)

    # calculate a background from the background file
    allvals = []
    sample_means = []
    for box in box_inds:
        rectangle = zip(box[0::2], box[1::2])
        sample = get_bg_sample(data, hdr, rectangle)
        for s in sample:
            allvals.append(s)
        sample_mean = np.nanmean(sample)
        sample_means.append(sample_mean)
    this_mean = np.around(np.nanmean(sample_means), 8)

    # subtract the background from the data
    final_data = data - this_mean

    # write the subtracted background file to the header for future use
    hdr['BG'] = this_mean
    hdr['comment'] = 'Background has been subtracted.'

    # write out the background-subtracted file
    outfile = os.path.join(final_dir, 'final_mosaic.fits')
    astropy.io.fits.writeto(outfile, final_data, hdr)


def read_bg_regfile(regfile):
    """
    Read the background region file (ds9 .reg file) and return the regions in which to calculate a background

    Parameters
    ----------
    regfile : str
        Path to file containing the regions in which to calculate a background

    Returns
    -------
    box_list : list
        List of data points that define the region in which to calculate a background
    """
    f = open(regfile, 'r')
    boxes = f.readlines()
    f.close()
    box_list = []
    for b in boxes:
        this_box = []
        box = b.strip('polygon()\n').split(',')
        [this_box.append(int(np.around(float(bb), 0))) for bb in box]
        box_list.append(this_box)
    return box_list


def get_bg_sample(data, hdr, box):
    """
    Determine the background sample in each reigon

    Parameters
    ----------
    data : float
        Input data
    hdr : FITS WCS
        header of input file containing WCS information
    box : np.2darray
        Numpy 2D array containing x,y coordinates of the box in which to determine the background

    Returns
    -------
    sample : float
        The data that lie within the region in which you want to calculate the background
    """
    wcs = astropy.wcs.WCS(hdr, naxis=2)
    x, y = np.arange(data.shape[0]), np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    xx, yy = X.flatten(), Y.flatten()
    pixels = np.array(zip(yy, xx))
    box_coords = box
    sel = Path(box_coords).contains_points(pixels)
    sample = data.flatten()[sel]
    return sample
