#!/usr/bin/env python
"""
Where you at?
"""
__author__  = "Alex Drlica-Wagner"
__email__   = "kadrlica@fnal.gov"
__version__ = "0.0.0"

"""
TODO:
- by default, show all images from current night
- clip bad data for previous nights
- 5-frame, 1 per band plots, including survey history
- (reach) upcoming exposures
- (reach) color sky brightness background
"""


import sys,os
import logging
from collections import OrderedDict as odict
from datetime import datetime,timedelta,tzinfo
import dateutil.parser

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Ellipse
#from matplotlib.patheffects import f
import numpy as np
import pylab as plt
import ephem

# For coloring filters
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

# For accessing footprints
FOOTPATH  = 'SISPI_FOOTPRINT'
FOOTPRINT = odict([
    ('des', 'round13-poly.txt'),
    ('none',None),
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

# Default maximum number of exposures to grab from DB
NMAX = 50000

# Stupid timezone definition
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
        query = "SELECT id,telra,teldec,filter FROM exposure WHERE exposed = TRUE AND flavor LIKE '%s' AND propid LIKE '%s' ORDER BY id DESC LIMIT %i"%(opts.flavor,opts.propid,NMAX)
        print query
        return np.rec.array(db.execute(query),dtype=dtype)
    else:
        return np.loadtxt(filename,dtype=dtype)

def load_footprint(footprint='des'):
    """ Load a footprint file """
    dtype=[('ra',float),('dec',float)]

    if footprint is None or footprint=='none':
        return np.array(len(dtype)*[[]],dtype=dtype)

    basedir  = os.path.dirname(os.path.abspath(__file__))
    default  = os.path.join(basedir,'..','data')
    dirname = os.environ.get(FOOTPATH,default)
    basename = FOOTPRINT[footprint]
    filename = os.path.join(dirname,basename)
    if not os.path.exists(filename):
        msg = "Footprint file not found: %s"%filename
        raise IOError(msg)
    return np.loadtxt(filename,dtype=dtype)

def lmst(datetime):
    """ Calculate Local Mean Sidereal Time (LMST) """
    lmst = np.degrees(CTIO.sidereal_time())
    logging.debug('Using pyephem for LMST: %.3f'%lmst)
    return lmst

def moon(datetime):
    """ Moon location """
    moon = ephem.Moon()
    moon.compute(CTIO)
    moon_phase = moon.moon_phase * 100
    moon_ra,moon_dec = np.degrees([moon.ra,moon.dec])
    return (moon_ra, moon_dec),moon_phase

def boolean(string):
    """ Convert strings to booleans for argparse """
    string = string.lower()
    if string in ['0', 'f', 'false', 'no', 'off']:
        return False
    elif string in ['1', 't', 'true', 'yes', 'on']:
        return True
    else:
        raise ValueError()

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('expnum',nargs='?',type=int,default=None,
                        help="exposure number to plot")
    parser.add_argument('-a','--airmass',default=1.4,type=float,
                        help='plot airmass limit')
    #parser.add_argument('--after',default=None,
    #                    help='plot exposures after a given UTC/ExpNum')
    parser.add_argument('-b','--band',default='all',choices=BANDS,
                        help='plot exposures in specific band')
    #parser.add_argument('--before',default=None,
    #                    help='plot exposures before a given UTC/ExpNum')
    parser.add_argument('-c','--color',default=True,type=boolean,
                        help='plot color corresponding to filter')
    parser.add_argument('-f','--footprint',default='des',choices=FOOTPRINT.keys(),
                        action='append',help='footprint to plot')
    parser.add_argument('--flavor',default='object',type=str,
                        help='exposure flavor [object,flat,etc.]')
    parser.add_argument('-i','--infile',default=None,
                        help='list of exposures to plot')
    parser.add_argument('-o','--outfile',default=None,
                        help='output file for saving figure')
    parser.add_argument('-m','--moon',default=True,type=boolean,
                        help='plot moon location and phase')
    parser.add_argument('-n','--numexp',default=10,type=float,
                        help='number of exposures to plot')
    parser.add_argument('--propid',default='%',
                        help='propid to filter exposures')
    parser.add_argument('--utc',default=None,
                        help="UTC for plot (defaults to now)")
    parser.add_argument('-v','--verbose',action='store_true',
                        help='verbosity')
    parser.add_argument('--version',action='version',version='%(prog)s '+__version__)
    parser.add_argument('-z','--zenith',default=True,type=boolean,
                        help="plot zenith position")

    opts = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=logging.DEBUG if opts.verbose else logging.INFO,
                        format='%(message)s',stream=sys.stdout)
    
    # Parse UTC
    if opts.utc is None:
        utc = datetime.now(tz=UTC())
    else: 
        utc = dateutil.parser.parse(opts.utc,tzinfos={'UTC':UTC})
    logging.debug("UTC: %s"%utc.strftime('%Y-%m-%d %H:%M:%S'))
    CTIO.date = utc

    # Grab the data
    data = load_data(opts.infile)
    
    # Subselect the data
    select = np.in1d(data['filter'],FILTERS)
    if opts.band in FILTERS:
        select &= (data['filter'] == opts.band)
    select &= (np.arange(len(data)) < opts.numexp)

    expnum,telra,teldec,band = data['expnum'],data['telra'],data['teldec'],data['filter']

    # Select the exposure of interest
    if opts.expnum:
        match = np.char.array(expnum).endswith(str(opts.expnum))
        if not match.any():
            msg = "Exposure matching %s not found"%opts.expnum
            raise ValueError(msg)
        idx = np.nonzero(match)[0][0]
    else:
        idx = 0

    # Set the colors
    if opts.color:
        nexp = len(expnum)
        ncolors = len(COLORS)
        color_repeat = np.repeat(COLORS.keys(),nexp).reshape(ncolors,nexp)
        color_idx = np.argmax(band==color_repeat,axis=0)
        color = np.array(COLORS.values())[color_idx]
    else:
        color = COLORS['none']

    # Create the figure
    fig,ax = plt.subplots(figsize=(12,8))
    #fig,ax = plt.subplots()

    # Create the Basemap
    lon_0 = lmst(utc);     lat_0 = TEL_LAT
    m = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0)
    parallels = np.arange(-90.,120.,30.)
    m.drawparallels(parallels)
    meridians = np.arange(0.,420.,60.)
    m.drawmeridians(meridians)
    for mer in meridians[:-1]:
        plt.annotate(r'$%i^{\circ}$'%mer,m(mer,5),ha='center')
    plt.annotate('East',xy=(1.02,0.5),ha='left',xycoords='axes fraction')
    plt.annotate('West',xy=(-.02,0.5),ha='right',xycoords='axes fraction')
    
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
    m.scatter(x[select],y[select],color=color[select],**nexp_kwargs)

    # Plot zenith position & focal plane scale
    zen_x,zen_y = m(lon_0,lat_0)
    zen_kwargs = dict(color='green',alpha=0.75,lw=1,zorder=0)
    if opts.zenith:
        logging.debug("Plotting zenith: (%.2f,%.2f)"%(lon_0,lat_0))
        m.plot(zen_x,zen_y,'+',ms=10,**zen_kwargs)
        logging.debug("Plotting focal plane scale.")
        m.tissot(lon_0, lat_0, 1.0, 100, fc='none', **zen_kwargs)
    
    # Plot airmass circle
    if not np.isnan(opts.airmass):
        logging.debug("Plotting airmass: %s"%opts.airmass)
        angle = airmass_angle(opts.airmass)
        m.tissot(lon_0, lat_0, angle, 100, fc='none',**zen_kwargs)

    # Moon location and phase
    if opts.moon:
        (moon_ra,moon_dec),moon_phase = moon(utc)
        logging.debug("Plotting moon: %i%%,(%.1f,%.1f)"%(moon_phase,moon_ra,moon_dec))
        moon_txt = '%i%%'%moon_phase
        moon_kwargs = dict(zorder=exp_zorder-1,fontsize=10,va='center',ha='center',
                           bbox=dict(boxstyle='circle,pad=0.4',fc='k',ec='k',alpha=0.25,lw=2))
                                      
        ax.annotate(moon_txt,m(moon_ra,moon_dec),**moon_kwargs)

    # Plot footprint(s) (should eventually be a loop over all footprints)
    ft_kwargs = dict(marker='o',mew=0,mfc='none',color='b',lw=2,zorder=exp_zorder-3)
    perim = load_footprint(opts.footprint)
    logging.debug("Plotting footprint: %s"%opts.footprint)
    proj = safe_proj(m,perim['ra'],perim['dec'])
    m.plot(*proj,**ft_kwargs)

    if opts.footprint == 'des':
        # Plot the SN fields
        logging.debug("Plotting supernova fields.")
         
        # This does the projection correctly, but fails at boundary
        sn_kwargs = dict(facecolor='none',edgecolor=ft_kwargs['color'],zorder=exp_zorder-1)
        # Check that point inside boundary
        fact = 0.99
        boundary = Ellipse((m.rmajor,m.rminor),2*(fact*m.rmajor),2*(fact*m.rminor))
                           
        for v in SN.values():
            if not boundary.contains_point(m(*v)): continue
            m.tissot(v[0],v[1],1.0,100,**sn_kwargs)
         
        # The SN labels
        sntxt_kwargs = dict(zorder=exp_zorder-1,fontsize=12,
                            bbox=dict(boxstyle='round,pad=0',fc='w',ec='none',
                                      alpha=0.25))
        for k,v in SN_LABELS.items():
            ax.annotate(k,m(*v),**sntxt_kwargs)

    # Annotate with some information
    logging.debug("Adding info text.")
    bbox_props = dict(boxstyle='round', facecolor='white')
    textstr= "%s %s\n"%("UTC:",utc.strftime('%Y-%m-%d %H:%M:%S'))
    textstr+="%s %i (%s)\n"%("Exposure:",expnum[idx],band[idx])
    textstr+="%s %i\n"%("NExp:",opts.numexp)
    textstr+="%s (%.1f$^{\circ}$,%.1f$^{\circ}$)\n"%("Zenith:",lon_0,lat_0)
    textstr+="%s %s\n"%("Airmass:",opts.airmass)
    textstr+="%s %i%% (%.1f$^{\circ}$,%.1f$^{\circ}$)\n"%("Moon:",moon_phase,moon_ra,moon_dec)
    textstr+="%s %s"%("Footprint:",opts.footprint)

    ax.annotate(textstr, xy=(0.98,0.98), xycoords='axes fraction',
                fontsize=10,ha='left',va='top', bbox=bbox_props)

    # Plot filter legend
    if opts.color:
        logging.debug("Adding filter legend.")
        leg_kwargs = dict(scatterpoints=1,fontsize=10,bbox_to_anchor=(0.08,0.20))
        handles, labels = [],[]
        for k in FILTERS:
            if k == 'VR':
                if not (band[select]=='VR').any() and not band[idx]=='VR': 
                    continue
            labels.append(k)
            handles.append(plt.scatter(None,None,color=COLORS[k],**exp_kwargs))
        plt.legend(handles,labels,**leg_kwargs)

    # Save the figure
    if opts.outfile:
        logging.debug("Saving figure to: %s"%opts.outfile)
        plt.savefig(opts.outfile)#,bbox_inches='tight')

    plt.show()
