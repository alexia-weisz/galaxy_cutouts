import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import os, sys, time
import numpy as np
from pdb import set_trace
import montage_wrapper as montage
import shutil
import gal_data
import config
import glob


#_TOP_DIR = '/data/tycho/0/leroy.42/allsky/'
#_INDEX_DIR = os.path.join(_TOP_DIR, 'z0mgs/')
_HOME_DIR = '/n/home00/lewis.1590/research/galbase_allsky/'
_DATA_DIR = '/n/home00/lewis.1590/research/galbase/gal_data/'
#_MOSAIC_DIR = os.path.join(_HOME_DIR, 'cutouts')

_GAL_DIR = os.path.join(_HOME_DIR, 'ngc2976')
_INPUT_DIR = os.path.join(_GAL_DIR, 'input')
_MOSAIC_DIR = os.path.join(_GAL_DIR, 'mosaics')


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create cutouts of a given size around each galaxy center.')
    parser.add_argument('--size', default=30.,help='cutout size in arcminutes')
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--copy', action='store_true')
    parser.add_argument('--convolve', action='store_true')
    parser.add_argument('--align', action='store_true')
    return parser.parse_args()

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

def convert_files(gal_dir, im_dir, wt_dir, band, fuv_toab, nuv_toab, pix_as):
    converted_dir = os.path.join(gal_dir, 'converted')
    os.makedirs(converted_dir)

    intfiles = sorted(glob.glob(os.path.join(im_dir, '*-int.fits')))
    wtfiles = sorted(glob.glob(os.path.join(wt_dir, '*-rrhr.fits')))

    int_outfiles = [os.path.join(converted_dir, os.path.basename(f).replace('.fits', '_mjysr.fits')) for f in intfiles]
    wt_outfiles = [os.path.join(converted_dir, os.path.basename(f)) for f in wtfiles]

    for i in range(len(intfiles)):
        if os.path.exists(wtfiles[i]):
            im, hdr = pyfits.getdata(intfiles[i], header=True)
            wt, whdr = pyfits.getdata(wtfiles[i], header=True)
            #wt = wtpersr(wt, pix_as)
            #if band.lower() == 'fuv':
            #    im = counts2jy_galex(im, fuv_toab, pix_as)
            if band.lower() == 'nuv':
                im = counts2jy_galex(im, nuv_toab, pix_as)
            if not os.path.exists(int_outfiles[i]):
            #    im -= np.mean(im)
                pyfits.writeto(int_outfiles[i], im, hdr)
            if not os.path.exists(wt_outfiles[i]):
                pyfits.writeto(wt_outfiles[i], wt, whdr)
        else:
            continue

    return converted_dir, converted_dir

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

def write_headerfile(header_file, header):
    f = open(header_file, 'w')
    for iii in range(len(header)):
        outline = str(header[iii:iii+1]).strip().rstrip('END').strip()+'\n'
        f.write(outline)
    f.close()

def mask_images(im_dir, wt_dir, gal_dir):
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

def mask_galex(intfile, wtfile, outfile=None, chip_rad = 1400, chip_x0=1920, chip_y0=1920, out_intfile=None, out_wtfile=None):

    if out_intfile is None:
        out_intfile = intfile.replace('.fits', '_masked.fits')
    if out_wtfile is None:
        out_wtfile = wtfile.replace('.fits', '_masked.fits')

    if not os.path.exists(out_intfile):
        data, hdr = pyfits.getdata(intfile, header=True)
        wt, whdr = pyfits.getdata(wtfile, header=True)
        #flag, fhdr = pyfits.getdata(flagfile, header=True)

        #factor = float(len(data)) / len(flag)
        #upflag = zoom(flag, factor, order=0)

        x = np.arange(data.shape[1]).reshape(1, -1) + 1
        y = np.arange(data.shape[0]).reshape(-1, 1) + 1
        r = np.sqrt((x - chip_x0)**2 + (y - chip_y0)**2)

        i = (r > chip_rad)
        j = (data == 0)
        k = (wt == -1.1e30)

        data = np.where(i | k, 0, data)  #0
        wt = np.where(i | k, 1e-20, wt) #1e-20

        pyfits.writeto(out_intfile, data, hdr)
        pyfits.writeto(out_wtfile, wt, whdr)

def reproject_images(template_header, input_dir, reprojected_dir, imtype, whole=False, exact=True, img_list=None):

    reproj_imtype_dir = os.path.join(reprojected_dir, imtype)
    os.makedirs(reproj_imtype_dir)

    input_table = os.path.join(input_dir, imtype + '_input.tbl')
    montage.mImgtbl(input_dir, input_table, corners=True, img_list=img_list)

    # Create reprojection directory, reproject, and get image metadata
    stats_table = os.path.join(reproj_imtype_dir, imtype+'_mProjExec_stats.log')
    montage.mProjExec(input_table, template_header, reproj_imtype_dir, stats_table, raw_dir=input_dir, whole=whole, exact=exact)

    reprojected_table = os.path.join(reproj_imtype_dir, imtype + '_reprojected.tbl')
    montage.mImgtbl(reproj_imtype_dir, reprojected_table, corners=True)

    return reproj_imtype_dir

def weight_images(im_dir, wt_dir, weight_dir):
    im_suff, wt_suff = '*_mjysr.fits', '*-rrhr.fits'
    imfiles = sorted(glob.glob(os.path.join(im_dir, im_suff)))
    wtfiles = sorted(glob.glob(os.path.join(wt_dir, wt_suff)))

    im_weight_dir = os.path.join(weight_dir, 'int')
    wt_weight_dir = os.path.join(weight_dir, 'rrhr')
    [os.makedirs(out_dir) for out_dir in [im_weight_dir, wt_weight_dir]]

    for i in range(len(imfiles)):
        imfile = imfiles[i]
        wtfile = wtfiles[i]
        im, hdr = pyfits.getdata(imfile, header=True)
        rrhr, rrhrhdr = pyfits.getdata(wtfile, header=True)

        # noise = 1. / np.sqrt(rrhr)
        # weight = 1 / noise**2
        wt = rrhr
        newim = im * wt

        #nf = imfiles[i].split('/')[-1].replace('.fits', '_weighted.fits')
        #newfile = os.path.join(weighted_dir, nf)
        newfile = os.path.join(im_weight_dir, os.path.basename(imfile))
        pyfits.writeto(newfile, newim, hdr)
        old_area_file = imfile.replace('.fits', '_area.fits')
        if os.path.exists(old_area_file):
            new_area_file = newfile.replace('.fits', '_area.fits')
            shutil.copy(old_area_file, new_area_file)

        #nf = wtfiles[i].split('/')[-1].replace('.fits', '_weights.fits')
        #weightfile = os.path.join(weights_dir, nf)
        weightfile = os.path.join(wt_weight_dir, os.path.basename(wtfile))
        pyfits.writeto(weightfile, wt, rrhrhdr)
        old_area_file = wtfile.replace('.fits', '_area.fits')
        if os.path.exists(old_area_file):
            new_area_file = weightfile.replace('.fits', '_area.fits')
            shutil.copy(old_area_file, new_area_file)

    return im_weight_dir, wt_weight_dir

def create_table(in_dir, dir_type=None):
    if dir_type is None:
        reprojected_table = os.path.join(in_dir, 'reprojected.tbl')
    else:
        reprojected_table = os.path.join(in_dir, dir_type + '_reprojected.tbl')
    montage.mImgtbl(in_dir, reprojected_table, corners=True)
    return reprojected_table

def coadd(template_header, output_dir, input_dir, output=None, add_type=None):
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
    image_file = os.path.join(output_dir, 'int_mosaic.fits')
    wt_file = os.path.join(output_dir, 'weights_mosaic.fits')
    count_file = os.path.join(output_dir, 'count_mosaic.fits')
    im, hdr = pyfits.getdata(image_file, header=True)
    wt = pyfits.getdata(wt_file)
    ct = pyfits.getdata(count_file)

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






def galex(band='fuv', ra_ctr=None, dec_ctr=None, size_deg=None, index=None, name=None):
    gal_dir = _GAL_DIR
    galaxy_mosaic_file = os.path.join(_MOSAIC_DIR, '_'.join([name, band]).upper() + '.FITS')

    start_time = time.time()

    # CALIBRATION FROM COUNTS TO ABMAG
    fuv_toab = 18.82
    nuv_toab = 20.08

    # PIXEL SCALE IN ARCSECONDS
    pix_as = 1.5  # galex pixel scale -- from galex docs

    # MAKE A HEADER
    pix_scale = 1.5 / 3600.  # 1.5 arbitrary: how should I set it?
    pix_len = size_deg / pix_scale
    target_hdr = create_hdr(ra_ctr, dec_ctr, pix_len, pix_scale)


    # SET UP THE OUTPUT
    ri_targ, di_targ = make_axes(target_hdr)
    sz_out = ri_targ.shape
    outim = ri_targ * np.nan
    prihdu = pyfits.PrimaryHDU(data=outim, header=target_hdr)
    target_hdr = prihdu.header


    # GATHER THE INPUT FILES
    #im_dir, wt_dir, nfiles = get_input(index, ind, data_dir, gal_dir)
    im_dir = _INPUT_DIR
    wt_dir = _INPUT_DIR

    # CONVERT INT FILES TO MJY/SR AND WRITE NEW FILES INTO TEMP DIR
    im_dir, wt_dir = convert_files(gal_dir, im_dir, wt_dir, band, fuv_toab, nuv_toab, pix_as)


    # APPEND UNIT INFORMATION TO THE NEW HEADER AND WRITE OUT HEADER FILE
    target_hdr['BUNIT'] = 'MJY/SR'
    hdr_file = os.path.join(gal_dir, name + '_template.hdr')
    write_headerfile(hdr_file, target_hdr)


    # MASK IMAGES
    im_dir, wt_dir = mask_images(im_dir, wt_dir, gal_dir)


    # REPROJECT IMAGES
    reprojected_dir = os.path.join(gal_dir, 'reprojected')
    os.makedirs(reprojected_dir)
    im_dir = reproject_images(hdr_file, im_dir, reprojected_dir, 'int')
    wt_dir = reproject_images(hdr_file, wt_dir, reprojected_dir,'rrhr')


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
    coadd(hdr_file, final_dir, wt_dir, output='weights')
    coadd(hdr_file, final_dir, im_dir, output='int')
    coadd(hdr_file, final_dir, im_dir, output='count',add_type='count')


    # DIVIDE OUT THE WEIGHTS
    imagefile = finish_weight(final_dir)


    # SUBTRACT OUT THE BACKGROUND
    #remove_background(final_dir, imagefile, bg_reg_file)


    # COPY MOSAIC FILES TO CUTOUTS DIRECTORY
    mos_files = ['image_mosaic.fits','weights_mosaic.fits','count_mosaic.fits']
    suffixes = ['.FITS', '_weight.FITS', '_count.FITS']

    for f, s in zip(mos_files, suffixes):
        shutil.copy(os.path.join(final_dir, f),
                    os.path.join(gal_dir, '_'.join([name, band]).upper() + s))


    # REMOVE GALAXY DIRECTORY AND EXTRA FILES
    #shutil.rmtree(gal_dir, ignore_errors=True)
    fdirs = [final_dir, weight_dir, reprojected_dir, os.path.join(gal_dir, 'converted'), os.path.join(gal_dir, 'masked')]

    for fdir in fdirs:
        shutil.rmtree(fdir, ignore_errors=True)

    # NOTE TIME TO FINISH
    stop_time = time.time()
    total_time = (stop_time - start_time) / 60.

    print total_time
    return


def main(**kwargs):

    if kwargs['cutout']:
        gals = gal_data.gal_data(tag='SINGS', data_dir=_DATA_DIR)
        n_gals = len(gals)
        size_deg = kwargs['size'] * 60. / 3600.

        for i in range(n_gals):
            this_gal = np.rec.fromarrays(gals[i], names=list(config.COLUMNS))
            galname = str(this_gal.name).replace(' ', '').upper()

            if galname == 'NGC2976':
                set_trace()
                galex(band='fuv', ra_ctr=this_gal.ra_deg, dec_ctr=this_gal.dec_deg, size_deg=size_deg, name=galname)

                #galex(band='nuv', ra_ctr=this_gal.ra_deg, dec_ctr=this_gal.dec_deg, size_deg=size_deg, name=galname)

    if kwargs['copy']:
        pass

    if kwargs['convolve']:
        pass

    if kwargs['align']:
        pass


if __name__ == '__main__':
    args = get_args()
    main(**vars(args))

