import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import os
import numpy as np
from pdb import set_trace
import montage_wrapper as montage
import shutil
import sys


_TOP_DIR = '/data/tycho/0/leroy.42/allsky/'
_INDEX_DIR = os.path.join(_TOP_DIR, 'code/')
_HOME_DIR = '/n/home00/lewis.1590/research/galbase_allsky/'
_MOSAIC_DIR = os.path.join(_HOME_DIR, 'cutouts')

def counts2jy(norm_mag, calibration_value, pix_as):
    # convert counts to Jy
    val = 10.**((norm_mag + calibration_value) / -2.5)
    val *= 3631.0
    # then to MJy
    val /= 1e6
    # then to MJy/sr
    val /= np.radians(pix_as / 3600.)**2
    return val


def counts2jy_galex(counts, cal, pix_as):
    # first convert to abmag
    abmag = -2.5 * np.log10(counts) + cal

    # then convert to Jy
    f_nu = 10**(abmag/-2.5) * 3631.

    # then to MJy
    f_nu *= 1e-6

    # then to MJy/sr
    val = f_nu / (np.radians(pix_as/3600))**2
    return val
    #val = flux / MJYSR2JYARCSEC / pixel_area / 1e-23 / C * FUV_LAMBDA**2


def calc_tile_overlap(ra_ctr, dec_ctr, pad=0.0, min_ra=0., max_ra=180., min_dec=-90., max_dec=90.):

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

    # PULL THE IMAGE/CUBE SIZES FROM THE HEADER
    naxis  = hdr['NAXIS']
    naxis1 = hdr['NAXIS1']
    naxis2 = hdr['NAXIS2']
    if naxis > 2:
        naxis3 = hdr['NAXIS3']

    ## EXTRACT FITS ASTROMETRY STRUCTURE
    ww = pywcs.WCS(hdr)

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
    raxis = np.squeeze(rimg[:, naxis2/2.])
    daxis = np.squeeze(dimg[naxis1/2., :])

    return rimg, dimg


def write_headerfile(header_file, header):
    f = open(header_file, 'w')
    for iii in range(len(header)):
        outline = str(header[iii:iii+1]).strip().rstrip('END').strip()+'\n'
        f.write(outline)
    f.close()


def reprojection(orig_file, im, old_hdr, new_hdr, data_dir, name=None):

    montage_dir = os.path.join(data_dir, '_montage')

    ## write im (in MJy/sr) to a file
    newfile = orig_file.split('/')[-1].replace('.fits', '_to_mjysr.fits')
    imfile = os.path.join(_HOME_DIR, newfile)
    if not os.path.exists(imfile):
        pyfits.writeto(imfile, im, old_hdr)
    #hdu = pyfits.PrimaryHDU(im, header=old_hdr)
    #hdu.writeto(imfile)

    # wrie new_hdr to a file
    hdr_file = os.path.join(_HOME_DIR, 'template.hdr')
    if name is not None:
        hdr_file = os.path.join(_HOME_DIR, name + '_template.hdr')
    write_headerfile(hdr_file, new_hdr)

    # reproject with new imfile and headerfile
    outfile = imfile.replace('.fits', '_reproj.fits')
    montage.mProject(imfile, outfile, hdr_file)
    #new_hdu = montage.reproject_hdu(hdu, header=hdr_file)

    #return outfile
    return new_hdu.data


def create_hdr(ra_ctr, dec_ctr, pix_len, pix_scale):
    hdr = pyfits.Header()
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


def unwise(band=1, ra_ctr=None, dec_ctr=None, size_deg=None, index=None, name=None):
    tel = 'unwise'
    data_dir = os.path.join(_TOP_DIR, tel, 'sorted_tiles')

    # READ THE INDEX FILE (IF NOT PASSED IN)
    if index is None:
        indexfile = os.path.join(_INDEX_DIR, tel + '_index_file.fits')
        ext = 1
        index, hdr = pyfits.getdata(indexfile, ext, header=True)

    # CALIBRATION TO GO FROM VEGAS TO ABMAG
    w1_vtoab = 2.683
    w2_vtoab = 3.319
    w3_vtoab = 5.242
    w4_vtoab = 6.604

    # NORMALIZATION OF UNITY IN VEGAS MAG
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
        im, hdr = pyfits.getdata(infile, header=True)
        ri, di = make_axes(hdr)

        hh = pywcs.WCS(target_hdr)
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
        im, new_hdr = pyfits.getdata(newimfile, header=True)

        useful = np.where(np.isfinite(im))
        outim[useful] = im[useful]

        return outim, target_hdr


def galex(band='fuv', ra_ctr=None, dec_ctr=None, size_deg=None, index=None, name=None):
    tel = 'galex'
    data_dir = os.path.join(_TOP_DIR, tel, 'sorted_tiles')
    problem_file = os.path.join(_HOME_DIR, 'problem_galaxies.txt')

    galaxy_mosaic_file = os.path.join(_MOSAIC_DIR, '_'.join([name, band]).upper() + '.FITS')

    #if not os.path.exists(galaxy_mosaic_file):
    if name:
        # READ THE INDEX FILE (IF NOT PASSED IN)
        if index is None:
            indexfile = os.path.join(_INDEX_DIR, tel + '_index_file.fits')
            ext = 1
            index, hdr = pyfits.getdata(indexfile, ext, header=True)

        # CALIBRATION FROM COUNTS TO ABMAG
        fuv_toab = 18.82
        nuv_toab = 20.08

        # PIXEL SCALE IN ARCSECONDS
        pix_as = 1.5  # galex pixel scale -- from galex docs

        # MAKE A HEADER
        pix_scale = 1.5 / 3600.  # 1.5 arbitrary: how should I set it?
        pix_len = size_deg / pix_scale
        target_hdr = create_hdr(ra_ctr, dec_ctr, pix_len, pix_scale)

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
        ct_overlap = len(ind[0])

        # MAKE SURE THERE ARE OVERLAPPING TILES
        if ct_overlap == 0:
            with open(problem_file, 'a') as myfile:
                myfile.write(name + ': ' + 'No overlapping tiles\n')
            return

        # SET UP THE OUTPUT
        ri_targ, di_targ = make_axes(target_hdr)
        sz_out = ri_targ.shape
        outim = ri_targ * np.nan
        prihdu = pyfits.PrimaryHDU(data=outim, header=target_hdr)
        target_hdr = prihdu.header

        # CREATE NEW DIRECTORY TO STORE TEMPORARY FILES
        gal_dir = os.path.join(_HOME_DIR, name)
        os.makedirs(gal_dir)

        infiles = infiles = index[ind[0]]['fname']
        infiles = [data_dir + '/' + f for f in infiles]

        # CREATE SUBDIRECTORY INSIDE TEMP DIRECTORY FOR THE INPUT FILES
        input_dir = os.path.join(gal_dir, 'input')
        os.makedirs(input_dir)
        outfiles = [os.path.join(input_dir, f.split('/')[-1].replace('.fits', '_mjysr.fits')) for f in infiles]

        # CONVERT TO MJY/SR AND WRITE NEW FILES INTO TEMPORARY DIRECTORY
        for i in range(len(infiles)):
            im, hdr = pyfits.getdata(infiles[i], header=True)
            if band.lower() == 'fuv':
                im = counts2jy_galex(im, fuv_toab, pix_as)
            if band.lower() == 'nuv':
                im = counts2jy_galex(im, nuv_toab, pix_as)
            if not os.path.exists(outfiles[i]):
                pyfits.writeto(outfiles[i], im, hdr)

        # APPEND UNIT INFORMATION TO THE NEW HEADER
        target_hdr['BUNIT'] = 'MJY/SR'

        # WRITE OUT A HEADER FILE
        hdr_file = os.path.join(gal_dir, name + '_template.hdr')
        write_headerfile(hdr_file, target_hdr)
        final_dir = os.path.join(gal_dir, 'mosaic')

        # MOSAIC THE TILES FOR THIS GALAXY
        try:
            montage.mosaic(input_dir, final_dir, header=hdr_file,background_match=False, combine='count')
            set_trace()
            # COPY MOSAIC FILE TO CUTOUTS DIRECTORY
            mosaic_file = os.path.join(final_dir, 'mosaic.fits')
            newfile = '_'.join([name, band]).upper() + '.FITS'
            new_mosaic_file = os.path.join(_MOSAIC_DIR, newfile)
            shutil.copy(mosaic_file, new_mosaic_file)

            # REMOVE GALAXY DIRECTORY AND EXTRA FILES
            shutil.rmtree(gal_dir)

        except Exception as inst:
            me = sys.exc_info()[0]
            problem_file = os.path.join(_HOME_DIR, 'problem_galaxies.txt')
            with open(problem_file, 'a') as myfile:
                myfile.write(name + ': ' + str(me) + ': ' + str(inst) + '\n')
            shutil.rmtree(gal_dir)

    return

