#!/usr/bin/env python
"""
Convert FITS binary table to CSV file.
"""
__author__ = "Alex Drlica-Wagner"

import os
from os.path import splitext, exists

import pandas as pd
import numpy as np
import fitsio

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile')
    parser.add_argument('outfile',nargs='?')
    parser.add_argument('-v','--verbose',action='store_true')
    args = parser.parse_args()

    base,ext = splitext(args.infile)

    try: 
        if args.verbose: print("Reading %s..."%args.infile)
        data = fitsio.read(args.infile)
    except:
        msg = "Input must be valid FITS file"
        raise Exception(msg)

    outfile = args.outfile
    if not outfile:
        outfile = base+'.csv'
        
    if exists(outfile): 
        if args.verbose: print("Removing %s..."%outfile)
        os.remove(outfile)
        
    if args.verbose: print("Writing %s..."%outfile)
    pd.DataFrame(data).to_csv(index=False)
    
    if args.verbose: print("Done.")
