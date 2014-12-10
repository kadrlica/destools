#!/usr/bin/env python
"""
Where you at?
"""
__author__ = "Alex Drlica-Wagner (kadrlica@fnal.gov)"
__version__ = "0.0.0"
__revision__ = ""

import logging
import sys,os
from collections import OrderedDict as odict
from datetime import datetime,timedelta,tzinfo
import dateutil.parser

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Ellipse
import numpy as np
import pylab as plt
import ephem

datadir = "/home3/data_local/images/fits/2012B-0001/"

FILTERS = ['u','g','r','i','z','Y','VR']
BANDS = FILTERS + ['all']
COLORS = odict([
    ('none','black'),
    ('u','blue'),
    ('g','green'),
    ('r','red'),
    ('i','gold'),
    ('z','magenta'),
    ('Y','black'),
    ('VR','gray'),
])

# Derived from telra,teldec of 10000 exposures
SN = odict([
    ('E1',(7.874, -43.010)),
    ('E2',(9.500, -43.999)),
    ('X1',(34.476, -4.931)),
    ('X2',(35.664,-6.413)),
    ('X3',(36.449, -4.601)),
    ('S1',(42.818, 0.000)),
    ('S2',(41.193, -0.991)),
    ('C1',(54.274, -27.113)),
    ('C2',(54.274, -29.090)),
    ('C3',(52.647, -28.101)),
])

SN_LABELS = odict([
    ('SN-E',(8,-41)),
    ('SN-X',(35,-12)),
    ('SN-S',(45,1)),
    ('SN-C',(55,-35)),
])

#http://www.ctio.noao.edu/noao/content/Coordinates-Observatories-Cerro-Tololo-and-Cerro-Pachon
#http://arxiv.org/pdf/1210.1616v3.pdf
#(-30 10 10.73, -70 48 23.52, 2213m)

TEL_LON = -70.80653
TEL_LAT = -30.169647
TEL_HEIGHT = 2213

CTIO = ephem.Observer()
CTIO.lon,CTIO.lat = str(TEL_LON),str(TEL_LAT)
CTIO.elevation = TEL_HEIGHT

NMAX = 10000

ZERO = timedelta(0)
HOUR = timedelta(hours=1)

class UTC(tzinfo):
    """UTC"""
    def utcoffset(self, dt):
        return ZERO

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return ZERO

def safe_proj(proj,lon,lat):
    """ Remove points outside of projection """
    x,y = proj(np.asarray(lon),np.asarray(lat))
    x[x > 1e29] = None
    y[y > 1e29] = None
    return x,y

def airmass_angle(x=1.4):
    """ Zenith angle for a given airmass limit """
    return 90.-np.degrees(np.arcsin(1./x))

def load_data(filename=None):
    """ Load the data (either from DB of file). """
    dtype=[('expnum',int),('telra',float),('teldec',float),('filter',object)]
    if filename is None:
        ### INSERT KLAUS' CODE HERE ###
        from database import Database
        db = Database()
        db.connect()
        query = "SELECT id,telra,teldec,filter FROM exposure WHERE exposed = TRUE AND flavor LIKE '%s' ORDER BY id DESC LIMIT %i"%(opts.flavor,NMAX)
        return np.rec.array(db.execute(query),dtype=dtype)
    else:
        return np.loadtxt(filename,dtype=dtype)

def mjd(datetime):
    mjd_epoch = dateutil.parser.parse('1858-11-17T00:00:00Z')
    mjd_date = (datetime-mjd_epoch).total_seconds()/float(24*60*60)
    return mjd_date

def lmst(datetime):
    """ Calculate Local Mean Sidereal Time (LMST) """
    try:
        ### THIS CAN BE USED IF ASTROPY UPDATED ###
        import astropy, astropy.time
        assert (astropy.version.minor >= 4)
        t = astropy.time.Time(datetime,scale='utc')
        t.delta_ut1_utc = t.get_delta_ut1_utc(return_status=True)[0]
        lmst = t.sidereal_time('mean',longitude=TEL_LON).degree
        logging.debug('Using astropy.time module for LMST: %.3f'%lmst)
        return lmst
    except:
        lmst = np.degrees(CTIO.sidereal_time())
        logging.debug('Using pyephem for LMST: %.3f'%lmst)
        return lmst
        ## http://star-www.rl.ac.uk/docs/sun67.htx/node118.html
        #from pyslalib import slalib
        #jd = mjd(utc)
        #t = slalib.sla_gmst(jd)
        #lmst = np.degrees(t)+TEL_LON
        #logging.debug('Using pyslalib for LMST: %.3f'%lmst)
        #return lmst

def moon(datetime):
    """ Moon location """
    ### from pyslalib import slalib
    ### lon_rad, lat_rad = np.radians([TEL_LON, TEL_LAT])
    ### mjd_date = mjd(datetime)
    ### moon_ra_rad, moon_dec_rad, d = slalib.sla_rdplan(mjd_date, 3, lat_rad, lon_rad)
    ### moon_ra, moon_dec = np.degrees([moon_ra_rad, moon_dec_rad])
    ### moon_phase = 0
    ###  
    ### print moon_ra,moon_dec
    ### print moon_phase

    moon = ephem.Moon()
    moon.compute(CTIO)
    moon_phase = moon.moon_phase * 100
    moon_ra,moon_dec = np.degrees([moon.ra,moon.dec])
    return (moon_ra, moon_dec),moon_phase


if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('expnum',nargs='?',type=int,default=None,
                        help="Exposure number to plot")
    parser.add_argument('-a','--airmass',default=1.4,type=float,
                        help='Airmass to plot')
    parser.add_argument('-b','--band',default='all',choices=BANDS,
                        help='Plot exposures in specific band.')
    parser.add_argument('-c','--color',default=True,type=bool,
                        help='Plot exposures in color corresponding to filter.')
    parser.add_argument('-f','--footprint',default='des',choices=['des','none'],
                        help='Footprint to plot')
    parser.add_argument('--flavor',default='object',type=str,
                        help='Exposure flavor [object,flat,etc.]')
    parser.add_argument('-i','--infile',default=None,
                        help='List of exposures to plot')
    parser.add_argument('-o','--outfile',default=None,
                        help='Output file for saving figure')
    parser.add_argument('-m','--moon',default=True,type=bool,
                        help='Plot moon location and phase')
    parser.add_argument('-n','--numexp',default=10,type=int,
                        help='Number of exposures to plot')
    parser.add_argument('--utc',default=None,
                        help="UTC for plot (defaults to now)")
    parser.add_argument('-v','--verbose',action='store_true',
                        help='Verbosity')
    parser.add_argument('-z','--zenith',type=bool,default=True,
                        help="Plot zenith position")

    opts = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=logging.DEBUG if opts.verbose else logging.INFO,
                        format='%(message)s',stream=sys.stdout)
    
    # Parse UTC
    if opts.utc is None:
        utc = datetime.now(tz=UTC())
    else: 
        utc = dateutil.parser.parse(opts.utc)
    logging.debug("UTC: %s"%utc.strftime('%Y-%m-%d %H:%M:%S'))
    CTIO.date = utc

    # Grab the data
    data = load_data(opts.infile)
    
    # Subselect the data
    sel = np.in1d(data['filter'],FILTERS)
    if opts.band in FILTERS:
        sel &= (data['filter'] == opts.band)
    data = data[sel]

    expnum,telra,teldec,band = data['expnum'],data['telra'],data['teldec'],data['filter']

    # Set the colors
    if opts.color:
        nexp = len(expnum)
        ncolors = len(COLORS)
        color_repeat = np.repeat(COLORS.keys(),nexp).reshape(ncolors,nexp)
        color_idx = np.argmax(band==color_repeat,axis=0)
        color = np.array(COLORS.values())[color_idx]
    else:
        color = COLORS['none']

    # Select the exposure of interest
    if opts.expnum:
        match = np.char.array(expnum).endswith(str(opts.expnum))
        if not match.any():
            msg = "Exposure matching %s not found"%opts.expnum
            raise ValueError(msg)
        idx = np.nonzero(match)[0][0]
    else:
        idx = 0

    # Create the figure
    fig,ax = plt.subplots()

    # Create the Basemap
    lon_0 = lmst(utc);     lat_0 = TEL_LAT
    m = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0)
    parallels = np.arange(-90.,120.,30.)
    m.drawparallels(parallels)
    meridians = np.arange(0.,420.,60.)
    m.drawmeridians(meridians)
    for mer in meridians[:-1]:
        plt.annotate(r'$%i^{\circ}$'%mer,m(mer,5),ha='center')
    plt.annotate('East',xy=(0.8,0.5),ha='center',xycoords='figure fraction')
    plt.annotate('West',xy=(0.22,0.5),ha='center',xycoords='figure fraction')
    
    exp_zorder = 10
    exp_kwargs = dict(s=40,marker='H',zorder=exp_zorder,edgecolor='k',lw=1)

    # Projected exposure locations
    x,y = safe_proj(m,telra,teldec)

    # Plot exposure of interest
    logging.debug("Plotting exposure: %i (%3.2f,%3.2f)"%(expnum[idx],telra[idx],teldec[idx]))
    m.scatter(x[idx],y[idx],color=color,**exp_kwargs)

    # Plot previous exposures
    nexp_kwargs = dict(exp_kwargs)
    nexp_kwargs.update(zorder=exp_zorder-1,alpha=0.2,edgecolor='none')#,lw=0)

    logging.debug("Plotting last %i exposures"%opts.numexp)
    m.scatter(x[:opts.numexp],y[:opts.numexp],color=color[:opts.numexp],**nexp_kwargs)

    # Plot zenith position
    zen_x,zen_y = m(lon_0,lat_0)
    zen_kwargs = dict(color='green',alpha=0.75,lw=1,zorder=0)
    if opts.zenith:
        logging.debug("Plotting zenith: (%.2f,%.2f)"%(lon_0,lat_0))
        m.plot(zen_x,zen_y,'+',ms=10,**zen_kwargs)
    
    # Plot airmass circle
    if not np.isnan(opts.airmass):
        logging.debug("Plotting airmass: %s"%opts.airmass)
        angle = airmass_angle(opts.airmass)
        m.tissot(lon_0, lat_0, angle, 100, fc='none',**zen_kwargs)

    # DES focal plane size
    if opts.verbose:
        logging.debug("Plotting focal plane scale.")
        m.tissot(lon_0, lat_0, 1.0, 100, fc='none', **zen_kwargs)

    # Moon location and phase
    if opts.moon:
        (moon_ra,moon_dec),moon_phase = moon(utc)
        logging.debug("Plotting moon: %i%%,(%.1f,%.1f)"%(moon_phase,moon_ra,moon_dec))
        moon_txt = '%i%%'%moon_phase
        moon_kwargs = dict(marker='o',fc='none',ec='k',mew=2)
        moon_kwargs = dict(zorder=exp_zorder-1,fontsize=10,va='center',ha='center',
                           bbox=dict(boxstyle='circle,pad=0.4',fc='k',ec='k',
                                      alpha=0.25,lw=2))
        ax.annotate(moon_txt,m(moon_ra,moon_dec),**moon_kwargs)

    # Plot footprint(s) (should eventually be a loop over all footprints)
    ft_kwargs = dict(marker='o',mew=0,mfc='none',color='b',lw=2,zorder=exp_zorder-3)
    if opts.footprint == 'des':
        # Plot the wide-field survey footprint
        logging.debug("Plotting footprint: %s"%opts.footprint)
        basedir = os.path.dirname(os.path.abspath(__file__))
        infile = os.path.join(basedir,'round13-poly.txt')
        perim = np.loadtxt(infile,
                           dtype=[('ra',float),('dec',float)])
        proj = safe_proj(m,perim['ra'],perim['dec'])
        m.plot(*proj,**ft_kwargs)

        # Plot the SN fields
        logging.debug("Plotting supernova fields.")

        # This does the projection correctly, but fails at boundary
        sn_kwargs = dict(facecolor='none',edgecolor=ft_kwargs['color'],zorder=exp_zorder-1)
        # Check that point inside boundary
        fact = 0.99
        boundary = Ellipse((m.rmajor,m.rminor),
                           2.*(fact*m.rmajor),2.*(fact*m.rminor))
        for v in SN.values():
            if not boundary.contains_point(m(*v)): continue
            m.tissot(v[0],v[1],1.0,100,**sn_kwargs)

        ### # This doesn't project, but is safe
        ### sn_kwargs = dict(marker='H',s=50,facecolor='none',edgecolor=ft_kwargs['color'],zorder=exp_zorder-1)
        ### m.scatter(*m(*zip(*SN.values())),**sn_kwargs)
        ###     

        # The SN labels
        sntxt_kwargs = dict(zorder=exp_zorder-1,fontsize=12,
                            bbox=dict(boxstyle='round,pad=0',fc='w',ec='none',
                                      alpha=0.25))
        for k,v in SN_LABELS.items():
            ax.annotate(k,m(*v),**sntxt_kwargs)

    # Annotation
    logging.debug("Adding info text.")
    bbox_props = dict(boxstyle='round', facecolor='white')
    textstr= "%s %s\n"%("UTC:",utc.strftime('%Y-%m-%d %H:%M:%S'))
    textstr+="%s %i (%s)\n"%("Exposure:",expnum[idx],band[idx])
    textstr+="%s %i\n"%("NExp:",opts.numexp)
    textstr+="%s (%.1f,%.1f)\n"%("Zenith:",lon_0,lat_0)
    textstr+="%s %s\n"%("Airmass:",opts.airmass)
    textstr+="%s %i%% (%.1f,%.1f)\n"%("Moon:",moon_phase,moon_ra,moon_dec)
    textstr+="%s %s"%("Footprint:",opts.footprint)

    ax.annotate(textstr, xy=(0.70,0.95), xycoords='figure fraction',
                fontsize=10,ha='left',va='top', bbox=bbox_props)

    # Plot filter legend
    if opts.band != 'none':
        logging.debug("Adding filter legend.")
        leg_kwargs = dict(scatterpoints=1,fontsize=10,
                          bbox_to_anchor=(0.28, 0.35),
                          bbox_transform=plt.gcf().transFigure)
        handles, labels = [],[]
        for k in FILTERS:
            if k == 'VR' and not (band=='VR').any(): continue
            labels.append(k)
            handles.append(plt.scatter(None,None,color=COLORS[k],**exp_kwargs))
        plt.legend(handles,labels,**leg_kwargs)

    # Save the figure
    if opts.outfile:
        logging.debug("Saving figure to: %s"%opts.outfile)
        plt.savefig(opts.outfile)#,bbox_inches='tight')

    plt.ion()



