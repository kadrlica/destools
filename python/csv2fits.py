#!/usr/bin/env python
"""
Convert CSV file to FITS binary table.
"""
__author__ = "Alex Drlica-Wagner"

from os.path import splitext

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
        data = pd.read_csv().to_records(index=False)
    except:
        msg = "Input must be valid CSV file"
        raise Exception(msg)

    names = [n.upper() for n in data.dtype.names]
    data.dtype.names = names

    outfile = args.outfile
    if not outfile:
        outfile = base+'.fits'
        
    if exists(outfile): 
        if args.verbose: print("Removing %s..."%outfile)
        os.remove(outfile)
        
    if args.verbose: print("Writing %s..."%outfile)
    fitsio.write(outfile,data)
    
    if args.verbose: print("Done.")
