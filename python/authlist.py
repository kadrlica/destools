#!/usr/bin/env python
"""
A simple script for making latex author lists from the csv file produced by 
the DES Publication Database.
"""
__author__ = "Alex Drlica-Wagner"
__email__ = "kadrlica@fnal.gov"

import csv
import numpy as np
import os
from collections import OrderedDict as odict

journal2class = odict([
    ('apj','aastex'),
    ('aj','aastex'),
    ('prl','revtex'),
    ('prd','revtex'),
    ('aastex','aastex'),
    ('revtex','revtex'),
])
defaults = dict(
    title = "DES Publication Title",
    abstract="This list is preliminary; the status is not yet ``ready to submit''.",
    collaboration="The DES Collaboration"
)

revtex_template = r"""
\documentclass[reprint,superscriptaddress]{revtex4-1}
\pagestyle{empty}
\begin{document}
\title{%(title)s}
 
%(authors)s

\collaboration{The DES Collaboration}

\begin{abstract}
%(abstract)s
\end{abstract}
\maketitle
\end{document}
"""

aastex_template = r"""
\documentclass[preprint]{aastex}
\pagestyle{empty}
\begin{document}
\title{%(title)s}
 
\author{
%(authors)s
\\ \vspace{0.2cm} (%(collaboration)s) \\
}
 
%(affiliations)s
 
\begin{abstract}
%(abstract)s
\end{abstract}
\maketitle
\end{document}
"""

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infile',metavar='DES-XXXX-XXXX_author_list.csv',
                        help="Input csv file from DES PubDB")
    parser.add_argument('outfile',metavar='DES-XXXX-XXXX_author_list.tex',
                        nargs='?',default=None,help="Output latex file (optional).")
    parser.add_argument('-f','--force',action='store_true',
                        help="Force overwrite of output.")
    parser.add_argument('-j','--journal',default='apj',choices=journal2class.keys(),
                        help="Journal name or latex document class.")
    parser.add_argument('-s','--sort',action='store_true',
                        help="Alphabetize the author list (you know you want to...).")
    parser.add_argument('-i','--idx',default=1,type=int,
                        help="Starting index for aastex author list (useful for mult-collaboration papers; better to use revtex).")
    opts = parser.parse_args()

    rows = [r for r in csv.reader(open(opts.infile)) if not r[0].startswith('#')]
    data = np.rec.fromrecords(rows[1:],names=rows[0])

    if opts.sort: data = data[np.argsort(np.char.upper(data['Lastname']))]

    affidict = odict()
    authdict = odict()
     
    if journal2class[opts.journal.lower()] == 'revtex':
        template = revtex_template

        for i,d in enumerate(data):
            #if d['Affiliation'] == '': continue
            if d['Authorname'] not in authdict.keys():
                authdict[d['Authorname']] = [d['Affiliation']]
            else:
                authdict[d['Authorname']].append(d['Affiliation'])

        authors = []
        for key,val in authdict.items():
            author = r'\author{%s}'%key+'\n'
            for v in val:
                author += r'\affiliation{%s}'%v+'\n'
            authors.append(author)
        params = dict(defaults,authors=''.join(authors))
        output = template%params

    if journal2class[opts.journal.lower()] == 'aastex':
        template = aastex_template
         
        for i,d in enumerate(data):
            #if d['Affiliation'] == '': continue
            if (d['Affiliation'] not in affidict.keys()):
                affidict[d['Affiliation']] = len(affidict.keys())
            affidx = affidict[d['Affiliation']]
            
            if d['Authorname'] not in authdict.keys():
                authdict[d['Authorname']] = [affidx]
            else:
                authdict[d['Authorname']].append(affidx)
         
        affiliations = []
        authors=[]
        for k,v in authdict.items():
            author = k+r'\altaffilmark{'+','.join([str(_v+opts.idx) for _v in v])+'}'
            authors.append(author)
         
        for k,v in affidict.items():
            affiliation = r'\altaffiltext{%i}{%s}'%(v+opts.idx,k)
            affiliations.append(affiliation)
            
        params = dict(defaults,authors=',\n'.join(authors),affiliations='\n'.join(affiliations))
        output = template%params
         

    if opts.outfile is not None:
        outfile = opts.outfile
    else:
        outfile = opts.infile.replace('.csv','.tex')
    if os.path.exists(outfile) and not opts.force:
        print "Found %s; skipping..."%outfile
    out = open(outfile,'w')
    out.write(output)
