import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import astropy.io.fits as pyfits
import os
import astropy.wcs as pywcs


def get_world(data, hdr):
	wcs = pywcs.WCS(hdr, naxis=2)
	x, y = np.arange(data.shape[0]), np.arange(data.shape[1])
	X, Y = np.meshgrid(x, y, indexing='ij')
	xx, yy = X.flatten(), Y.flatten()
	pixels = np.array(zip(yy,xx))
	world = wcs.wcs_pix2world(pixels, 1)
	wra = world[:,0].reshape(data.shape[0], data.shape[1])
	wdec = world[:,1].reshape(data.shape[0], data.shape[1])
	#extent = [np.max(world[:,0]), np.min(world[:,0]), np.min(world[:,1]), np.max(world[:,1])]
	extent = [wra[0,0], wra[-1,-1], wdec[0,0], wdec[-1,-1]]
	return world, extent, wcs

SAVE = True
_TOP_DIR = '/Users/lewis.1590/research/z0mgs'
galaxies = ['NGC0300', 'NGC1682', 'NGC2976', 'NGC6744', 'NGC7800']
gals_pgc = ['PGC3238', 'PGC16211', 'PGC28120', 'PGC62836', 'PGC73177']

nuv_vmin = {'NGC0300':-2.5, 'NGC1682':-2.5, 'NGC2976':-2.5, 'NGC6744':-2.5, 'NGC7800':-3.0}
nuv_vmax = {'NGC0300':-0.5, 'NGC1682':-0.5, 'NGC2976':-0.5, 'NGC6744':-0.5, 'NGC7800':-1.0}

for gal, pgc in zip(galaxies, gals_pgc):

	_WORK_DIR = os.path.join(_TOP_DIR, 'examples')

	infile = os.path.join(_WORK_DIR, 'more_pt_sources_{}.csv'.format(gal.lower()))
	nuvfile = os.path.join(_WORK_DIR, '{}_NUV.FITS'.format(gal))
	fuvfile = os.path.join(_WORK_DIR, '{}_FUV.FITS'.format(gal))
	w1file = os.path.join(_WORK_DIR, 'wise', '{}_w1_mjysr.fits'.format(pgc))
	w2file = os.path.join(_WORK_DIR, 'wise', '{}_w2_gauss15.fits'.format(pgc))


	data = pandas.read_csv(infile, header=0)
	nuv, nuvhdr = pyfits.getdata(nuvfile, header=True)
	fuv, fuvhdr = pyfits.getdata(fuvfile, header=True)
	w1, w1hdr = pyfits.getdata(w1file, header=True)
	w2, w2hdr = pyfits.getdata(w2file, header=True)


	nuvworld, nuvextent, nuvwcs = get_world(nuv, nuvhdr)
	fuvworld, fuvextent, fuvwcs = get_world(fuv, fuvhdr)
	w1world, w1extent, w1wcs = get_world(w1, w1hdr)
	w2world, w2extent, w2wcs = get_world(w2, w2hdr)

	cut_nuv = (data.nuv_mag > -999.)
	cut_fuv = (data.fuv_mag > -999.)
	cut1 = (data.nuv_mag > -999.) & (data.nuv_mag <= 18)
	cut2 = (data.nuv_mag > 18.) & (data.nuv_mag <= 20)
	cut3 = (data.nuv_mag > 20.) & (data.nuv_mag <= 22)
	cut4 = (data.nuv_mag > 22.) & (data.nuv_mag <= 24)
	cut5 = (data.nuv_mag > 24.) & (data.nuv_mag <= 26)

	cmap1 = plt.cm.gray
	cmap2 = plt.cm.plasma

	pt_kwargs = {'marker': 'o', 'lw': 0, 'ms': 2}
	im_kwargs = {'cmap': cmap1, 'origin': 'lower', 'aspect': 'auto'}#, 'extent': nuvextent}

	labels = ['NUV', 'W1', 'W2']

	#fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(7, 9))
	fig = plt.figure(figsize=(7,9))
	ax1 = fig.add_subplot(3,2,1, projection=nuvwcs)
	ax2 = fig.add_subplot(3,2,2, projection=nuvwcs)
	ax3 = fig.add_subplot(3,2,3, projection=w1wcs)
	ax4 = fig.add_subplot(3,2,4, projection=w1wcs)
	ax5 = fig.add_subplot(3,2,5, projection=w2wcs)
	ax6 = fig.add_subplot(3,2,6, projection=w2wcs)

	ax1.imshow(np.log10(nuv), vmin=nuv_vmin[gal], vmax=nuv_vmax[gal], **im_kwargs)
	ax2.imshow(np.log10(nuv), vmin=nuv_vmin[gal], vmax=nuv_vmax[gal], **im_kwargs)

	ax3.imshow(w1, vmin=-0.01, vmax=0.1, **im_kwargs)
	ax4.imshow(w1, vmin=-0.01, vmax=0.1, **im_kwargs)

	ax5.imshow(w2, vmin=-0.01, vmax=0.1, **im_kwargs)
	ax6.imshow(w2, vmin=-0.01, vmax=0.1, **im_kwargs)

	for ax in [ax2, ax4, ax6]:
		ptax = ax.scatter(data.ra[cut_nuv], data.dec[cut_nuv], c=data.nuv_mag[cut_nuv], 
						  marker='o', s=2, cmap=cmap2, transform=ax.get_transform('fk5'))

	for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
		ra, dec = ax.coords['ra'], ax.coords['dec']
		ra.set_major_formatter('d.d')
		dec.set_major_formatter('d.d')
		ra.set_ticklabel_visible(False)
		dec.set_ticklabel_visible(False)
		ax.coords.grid(color='0.7', alpha=0.4, linestyle='-', linewidth=0.5)
		#ax.grid(color='0.7', linestyle='-', linewidth=0.5, alpha=0.3)
		#ax.tick_params(axis='both', labelbottom='off', labelleft='off')

	for i, ax in enumerate([ax1, ax3, ax5]):
		txt = ax.text(0.05, 0.9, r'\textbf{' + labels[i] + '}', size=18, color='#F3363E', transform=ax.transAxes)
		txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='0.2')])

	plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.02, right=0.85, bottom=0.02, top=0.95)
	pos2 = ax2.get_position()
	pos6 = ax6.get_position()

	cax = fig.add_axes([pos6.x1 + 0.01, pos6.y0, 0.04, pos2.y1-pos6.y0])
	cb = fig.colorbar(ptax, cax=cax, orientation='vertical')
	cb.ax.yaxis.set_ticks_position('right')
	cb.ax.tick_params(labelsize=16)
	cb.set_label('NUV mag', size=18, labelpad=6)

	plt.suptitle('{} ({})'.format(gal, pgc), size=22)

	if SAVE:
		plotname = os.path.join(_WORK_DIR, '{}.pdf'.format(gal))
		plt.savefig(plotname, dpi=300, bbox_inches='tight', format='pdf')
		plt.close()
	else:
		plt.show()




