import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import os
import numpy as np
import montage_wrapper as montage
import shutil
import sys
import glob
import time
from matplotlib.path import Path
from scipy.ndimage import zoom
from pdb import set_trace


_TOP_DIR = '/data/tycho/0/leroy.42/allsky/'
_INDEX_DIR = os.path.join(_TOP_DIR, 'code/')
_HOME_DIR = '/n/home00/lewis.1590/research/galbase_allsky/'
_MOSAIC_DIR = os.path.join(_HOME_DIR, 'cutouts')


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
    raxis = np.squeeze(rimg[:, naxis2/2])
    daxis = np.squeeze(dimg[naxis1/2, :])

    return rimg, dimg


def write_headerfile(header_file, header):
    f = open(header_file, 'w')
    for iii in range(len(header)):
        outline = str(header[iii:iii+1]).strip().rstrip('END').strip()+'\n'
        f.write(outline)
    f.close()


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




def unwise(band=None, ra_ctr=None, dec_ctr=None, size_deg=None, index=None, name=None):
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


def counts2jy(norm_mag, calibration_value, pix_as):
    # convert counts to Jy
    val = 10.**((norm_mag + calibration_value) / -2.5)
    val *= 3631.0
    # then to MJy
    val /= 1e6
    # then to MJy/sr
    val /= np.radians(pix_as / 3600.)**2
    return val




def galex(band='fuv', ra_ctr=None, dec_ctr=None, size_deg=None, index=None, name=None, write_info=True):
    tel = 'galex'
    data_dir = os.path.join(_TOP_DIR, tel, 'sorted_tiles')
    problem_file = os.path.join(_HOME_DIR, 'problem_galaxies.txt')
    #numbers_file = os.path.join(_HOME_DIR, 'number_of_tiles_per_galaxy.dat')
    bg_reg_file = os.path.join(_HOME_DIR, 'galex_reprojected_bg.reg')
    numbers_file = os.path.join(_HOME_DIR, 'gal_reproj_info.dat')

    galaxy_mosaic_file = os.path.join(_MOSAIC_DIR, '_'.join([name, band]).upper() + '.FITS')

    start_time = time.time()
    #if not os.path.exists(galaxy_mosaic_file):
    if name == 'NGC2976':
        print name
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
            if write_info:
                with open(problem_file, 'a') as myfile:
                    myfile.write(name + ': ' + 'No overlapping tiles\n')
            return

        # SET UP THE OUTPUT
        ri_targ, di_targ = make_axes(target_hdr)
        sz_out = ri_targ.shape
        outim = ri_targ * np.nan
        prihdu = pyfits.PrimaryHDU(data=outim, header=target_hdr)
        target_hdr = prihdu.header

        # GATHER THE INPUT FILES
        infiles = index[ind[0]]['fname']
        wtfiles = index[ind[0]]['rrhrfile']
        flgfiles = index[ind[0]]['flagfile']
        infiles = [os.path.join(data_dir, f) for f in infiles]
        wtfiles = [os.path.join(data_dir, f) for f in wtfiles]
        flgfiles = [os.path.join(data_dir, f) for f in flgfiles]

        # CREATE NEW TEMP DIRECTORY TO STORE TEMPORARY FILES
        gal_dir = os.path.join(_HOME_DIR, name)
        os.makedirs(gal_dir)

        # CREATE SUBDIRECTORIES INSIDE TEMP DIRECTORY FOR ALL TEMP FILES
        input_dir = os.path.join(gal_dir, 'input')
        reprojected_dir = os.path.join(gal_dir, 'reprojected')
        weights_dir = os.path.join(gal_dir, 'weights')
        weighted_dir = os.path.join(gal_dir, 'weighted')
        final_dir = os.path.join(gal_dir, 'mosaic')

        for indir in [input_dir, reprojected_dir, weights_dir, weighted_dir, final_dir]:
            os.makedirs(indir)

        # SYMLINK ORIGINAL RRHR FILES TO TEMPORARY INPUT DIRECTORY
        for wtfile in wtfiles:
            basename = os.path.basename(wtfile)
            new_wt_file = os.path.join(input_dir, basename)
            os.symlink(wtfile, new_wt_file)

        for flgfile in flgfiles:
            basename = os.path.basename(flgfile)
            new_flg_file = os.path.join(input_dir, basename)
            os.symlink(flgfile, new_flg_file)

        # CONVERT INT FILES TO MJY/SR AND WRITE NEW FILES INTO TEMP DIR
        # CONVERT WT FILES TO WT/SR AND WRITE NEW FILES INTO TEMP DIR
        int_outfiles = [os.path.join(input_dir, f.split('/')[-1].replace('.fits', '_mjysr.fits')) for f in infiles]
        wt_outfiles = [os.path.join(input_dir, f.split('/')[-1].replace('.fits', '_sr.fits')) for f in wtfiles]
        for i in range(len(infiles)):
            im, hdr = pyfits.getdata(infiles[i], header=True)
            wt, whdr = pyfits.getdata(wtfiles[i], header=True)
            wt = wtpersr(wt, pix_as)
            if band.lower() == 'fuv':
                im = counts2jy_galex(im, fuv_toab, pix_as)
            if band.lower() == 'nuv':
                im = counts2jy_galex(im, nuv_toab, pix_as)
            if not os.path.exists(int_outfiles[i]):
                pyfits.writeto(int_outfiles[i], im, hdr)
                pyfits.writeto(wt_outfiles[i], wt, whdr)

        # APPEND UNIT INFORMATION TO THE NEW HEADER
        target_hdr['BUNIT'] = 'MJY/SR'

        # WRITE OUT A HEADER FILE
        hdr_file = os.path.join(gal_dir, name + '_template.hdr')
        write_headerfile(hdr_file, target_hdr)

        # PERFORM THE REPROJECTION, WEIGHTING, AND EXTRACTION
        #try:
        # REPROJECT INPUT IMAGES (-int and -rrhr)
        int_suff, rrhr_suff, flag_suff = '*_mjysr.fits', '*-rrhr_sr.fits', '*-flags.fits'
        int_images = sorted(glob.glob(os.path.join(input_dir, int_suff)))
        rrhr_images = sorted(glob.glob(os.path.join(input_dir, rrhr_suff)))
        flag_images = sorted(glob.glob(os.path.join(input_dir, flag_suff)))
        reproject_images(hdr_file, int_images, rrhr_images, flag_images, input_dir, reprojected_dir)

        # WEIGHT IMAGES
        im_suff, wt_suff = '*_mjysr_masked.fits', '*-rrhr_sr_masked.fits'
        imfiles = sorted(glob.glob(os.path.join(reprojected_dir, im_suff)))
        wtfiles = sorted(glob.glob(os.path.join(reprojected_dir, wt_suff)))
        weight_images(imfiles, wtfiles, weighted_dir, weights_dir)

        # CREATE THE METADATA TABLES NEEDED FOR COADDITION
        tables = create_tables(weights_dir, weighted_dir)

        # COADD THE REPROJECTED, WEIGHTED IMAGES AND THE WEIGHT IMAGES
        coadd(hdr_file, final_dir, weights_dir, weighted_dir)

        # DIVIDE OUT THE WEIGHTS
        imagefile = finish_weight(final_dir)

        # SUBTRACT OUT THE BACKGROUND
        remove_background(final_dir, imagefile, bg_reg_file)


        # COPY MOSAIC FILE TO CUTOUTS DIRECTORY
        mosaic_file = os.path.join(final_dir, 'final_mosaic.fits')
        newfile = '_'.join([name, band]).upper() + '.FITS'
        new_mosaic_file = os.path.join(_MOSAIC_DIR, newfile)
        shutil.copy(mosaic_file, new_mosaic_file)

        # REMOVE GALAXY DIRECTORY AND EXTRA FILES
        #shutil.rmtree(gal_dir)
        stop_time = time.time()

        total_time = (stop_time - start_time) / 60.

        # WRITE OUT THE NUMBER OF TILES THAT OVERLAP THE GIVEN GALAXY
        if write_info:
            out_arr = [name, len(infiles), np.around(total_time,2)]
            with open(numbers_file, 'a') as nfile:
                nfile.write('{0: >10}'.format(out_arr[0]))
                nfile.write('{0: >6}'.format(out_arr[1]))
                nfile.write('{0: >6}'.format(out_arr[2]) + '\n')
                #nfile.write(name + ': ' + str(len(infiles)) + '\n')


        # SOMETHING WENT WRONG
        #except Exception as inst:
        #    me = sys.exc_info()[0]
        #    if write_info:
        #        with open(problem_file, 'a') as myfile:
        #            myfile.write(name + ': ' + str(me) + ': '+str(inst)+'\n')
        #    shutil.rmtree(gal_dir)

    return

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


def wtpersr(wt, pix_as):
    return wt / (np.radians(pix_as/3600))**2


def mask_galex(intfile, wtfile, flagfile, outfile=None, chip_rad = 1400, chip_x0=1920, chip_y0=1920, out_intfile=None, out_wtfile=None):

    if out_intfile is None:
        out_intfile = intfile.replace('.fits', '_masked.fits')
    if out_wtfile is None:
        out_wtfile = wtfile.replace('.fits', '_masked.fits')

    if not os.path.exists(out_intfile):
        data, hdr = pyfits.getdata(intfile, header=True)
        wt, whdr = pyfits.getdata(wtfile, header=True)
        flag, fhdr = pyfits.getdata(flagfile, header=True)

        factor = float(len(data)) / len(flag)
        upflag = zoom(flag, factor, order=0)

       # chip_x0, chip_y0  = hdr['CRPIX1'], hdr['CRPIX2']
        x = np.arange(data.shape[1]).reshape(1, -1) + 1
        y = np.arange(data.shape[0]).reshape(-1, 1) + 1
        r = np.sqrt((x - chip_x0)**2 + (y - chip_y0)**2)
        i = (r > chip_rad) | (data == 0)
        data = np.where(i, 0, data)
        wt = np.where(i, 1e-20, wt)
        pyfits.writeto(out_intfile, data, hdr)
        pyfits.writeto(out_wtfile, wt, whdr)


def reproject_images(template_header, int_images, rrhr_images, flag_images, input_dir, reprojected_dir, whole=True, exact=True):

    # MASK IMAGES
    for i in range(len(int_images)):
        image_infile = int_images[i]
        wt_infile = rrhr_images[i]
        flg_infile = flag_images[i]

        image_outfile = os.path.join(input_dir, os.path.basename(image_infile).replace('.fits', '_masked.fits'))
        wt_outfile = os.path.join(input_dir, os.path.basename(wt_infile).replace('.fits', '_masked.fits'))

        mask_galex(image_infile, wt_infile, flg_infile, out_intfile=image_outfile, out_wtfile=wt_outfile)

    # REPROJECT IMAGES
    input_table = os.path.join(input_dir, 'input.tbl')
    montage.mImgtbl(input_dir, input_table, corners=True)

    # Create reprojection directory, reproject, and get image metadata
    stats_table = os.path.join(reprojected_dir, 'mProjExec_stats.log')

    montage.mProjExec(input_table, template_header, reprojected_dir, stats_table, raw_dir=input_dir, whole=whole, exact=exact)

    reprojected_table = os.path.join(reprojected_dir, 'reprojected.tbl')
    montage.mImgtbl(reprojected_dir, reprojected_table, corners=True)


def weight_images(imfiles, wtfiles, weighted_dir, weights_dir):
    for i in range(len(imfiles)):
        imfile = imfiles[i]
        wtfile = wtfiles[i]
        im, hdr = pyfits.getdata(imfile, header=True)
        rrhr, rrhrhdr = pyfits.getdata(wtfile, header=True)

        wt = rrhr
        newim = im * wt

        nf = imfiles[i].split('/')[-1].replace('.fits', '_weighted.fits')
        newfile = os.path.join(weighted_dir, nf)
        pyfits.writeto(newfile, newim, hdr)
        old_area_file = imfiles[i].replace('.fits', '_area.fits')
        new_area_file = newfile.replace('.fits', '_area.fits')
        shutil.copy(old_area_file, new_area_file)

        nf = wtfiles[i].split('/')[-1].replace('.fits', '_weights.fits')
        weightfile = os.path.join(weights_dir, nf)
        pyfits.writeto(weightfile, wt, rrhrhdr)
        old_area_file = wtfiles[i].replace('.fits', '_area.fits')
        new_area_file = weightfile.replace('.fits', '_area.fits')
        shutil.copy(old_area_file, new_area_file)


def create_tables(weights_dir, weighted_dir):
    return_tables = []
    in_dir = weights_dir
    reprojected_table = os.path.join(in_dir, 'weights_reprojected.tbl')
    montage.mImgtbl(in_dir, reprojected_table, corners=True)
    return_tables.append(reprojected_table)

    in_dir = weighted_dir
    reprojected_table = os.path.join(in_dir, 'int_reprojected.tbl')
    montage.mImgtbl(in_dir, reprojected_table, corners=True)
    return_tables.append(reprojected_table)

    return return_tables


def coadd(template_header, output_dir, weights_dir, weighted_dir):
    img_dirs = [weights_dir, weighted_dir]
    outputs = ['weights', 'int']

    for img_dir, output in zip(img_dirs, outputs):
        reprojected_table = os.path.join(img_dir, output + '_reprojected.tbl')
        out_image = os.path.join(output_dir, output + '_mosaic.fits')
        montage.mAdd(reprojected_table, template_header, out_image, img_dir=img_dir, exact=True)


def finish_weight(output_dir):
    image_file = os.path.join(output_dir, 'int_mosaic.fits')
    wt_file = os.path.join(output_dir, 'weights_mosaic.fits')
    im, hdr = pyfits.getdata(image_file, header=True)
    wt, wthdr = pyfits.getdata(wt_file, header=True)

    newim = im / wt

    newfile = os.path.join(output_dir, 'image_mosaic.fits')
    pyfits.writeto(newfile, newim, hdr)
    return newfile


def remove_background(final_dir, imfile, bgfile):
    data, hdr = pyfits.getdata(imfile, header=True)
    box_inds = read_bg_regfile(bgfile)
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

    final_data = data - this_mean
    hdr['BG'] = this_mean
    hdr['comment'] = 'Background has been subtracted.'

    outfile = os.path.join(final_dir, 'final_mosaic.fits')
    pyfits.writeto(outfile, final_data, hdr)


def read_bg_regfile(regfile):
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
    wcs = pywcs.WCS(hdr, naxis=2)
    x, y = np.arange(data.shape[0]), np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    xx, yy = X.flatten(), Y.flatten()
    pixels = np.array(zip(yy, xx))
    box_coords = box
    sel = Path(box_coords).contains_points(pixels)
    sample = data.flatten()[sel]
    return sample
