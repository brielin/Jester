#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from jester import sample
from jester import fit
from scipy import stats
from math import exp
from IPython import embed
import time
import numpy as np
import scipy as sp
import pandas as pd
import sys
import argparse


__version__='2.0.0'
header='Jester version '+__version__+'\n'\
    '(C) 2014-2016 Brielin C Brown\n'\
    'University of California, Berkeley\n'\
    'GNU General Public License v3\n'

def main(argv):
    # Arguments for both sample and test mode
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',help='Specify output verbosity',type=int,
                        choices=[0,1,2,3])
    parser.add_argument('--out',help='Specify output file',default='jester.out')
    parser.add_argument('--maf',help='Specify MAF cutoff',type=float,
                        default=0.05)
    parser.add_argument("--bfile", help="Base filename, without"
                        " extension", default=None)
    parser.add_argument("--covfile", default=None,
                        help="The location of a plink format covariate file")
    parser.add_argument("--r2min", help="Sets the minimum "
                        "squared corrleation required for joint testing."
                        "(Default = 0.0) ", default=0.0, type=float)
    # parser.add_argument("--r2Max", dest = "rMax", help="Sets the maximum "
    #                     "squared correletion for joint test. (Default = 1.0)",
    #                     default=1.0, type=float)
    # parser.add_argument("-L", dest = "L", help="Sets the length of analysis. "
    #                     "Default is whole input.", default=0, type=int)
    parser.add_argument('--from_bp',default=None,type=int)
    parser.add_argument('--to_bp',default=None,type=int)
    parser.add_argument('--SNPs_to_read',help='Specify number of SNPs'
                        'to read at a time. Do not set yourself',
                        default=1000,type=int)
    parser.add_argument('--SNPs_to_store',help='Specify size of'
                        'in memory SNP array. May need to increase'
                        'for dense panels.',type=int,default=2000)
    # parser.add_argument('--window_type',help='Specify window type'
    #                     ' (SNP,BP)',default='SNP')
    parser.add_argument("--window_size", help="Sets the sliding "
                        "window size. (Default = 100)",
                        default=100, type=int)
    parser.add_argument('--extract',default=None)
    subparsers = parser.add_subparsers(help='Program mode: "sample" for '
                                       'sampling from the null, "fit" for '
                                       'fitting joint model', dest='mode')
    # Arguments for sample mode
    parser_sample = subparsers.add_parser('sample')
    parser_sample.add_argument("--alpha",help="Desired level-alpha sig"
                               "nificance threashold (Default = 0.05)",
                               default=0.05,type=float)
    parser_sample.add_argument("--numSamples", help="Number of"
                               " samples. (Default = 1000)",
                               default=1000, type=int)
    parser_sample.add_argument("--sample_window_size",help="Window size for"
                               " sampling MVN. Default is same as joint testing"
                               " window size",default=None)
    # parser_sample.add_argument("--sample_window_type",help="Window type for"
    #                            " sampling MVN. Detault is same as joint testing"
    #                            " window type",default=None)
    parser_sample.add_argument("--seed",help="Sets the numpy seed. Default"
                        " is unset.",default=0,type=int)
    # Arguments for fit mode
    parser_test = subparsers.add_parser('fit')
    parser_test.add_argument("--phenofile", default=None,
                             help="Specify phenotype file location. Without this"
                             " argument the program will look for a .pheno that "
                             " has the plinkFileBase root")
    parser_test.add_argument("--minp",help="Sets the minimum expected"
                             "p-value for doing the joint test. Defult is 1e-05. Set"
                             "to 1 to fit the model for every pair.",
                             default=1e-05,type=float)
    parser_test.add_argument("--linear",help="Use linear regression,"
                             " even if the phenotype is binary.",default=False,
                             action="store_true")
    args = parser.parse_args()
    args.window_type = 'SNP'

    print(header)
    print('Invoking command: python '+' '.join(sys.argv))
    print('Beginning analysis at {T}'.format(T=time.ctime()))
    start_time = time.time()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    if (not args.bfile) or (not args.out):
        parser.error("You must provide a base file, and output file")

    if args.mode == "sample":
        args.sample_window_size = args.sample_window_size or args.window_size
        args.sample_window_type = 'SNP'
        if args.seed != 0: np.random.seed(args.seed)
        sample_res = sample.sample(args)
        # sample_res.ZMinP.to_pickle( args.out + '.Z.pkl' )
        # sample_res.JMinP.to_pickle( args.out + '.J.pkl' )
        index = int(np.round((args.alpha)*(args.numSamples-1)))
        alphaC_Z = np.sort(sample_res.ZMinP)[index]
        alphaC_J = np.sort(sample_res.minP)[index]
        print("Approximate level-{a} significance threshold:".format(a=args.alpha))
        print("Marginal: {a}".format(a=alphaC_Z))
        print("Joint: {a}".format(a=alphaC_J))
    elif args.mode == "fit":
        if (args.minp > 1.0) or (args.minp <= 0.0):
            parser.error("minp must be between 0 and 1")
        fit_res = fit.fit(args)
        fit_res.joint_res.to_csv(args.out+'.joint.out',sep='\t',na_rep='NA',
                                 float_format='%.5g')
        fit_res.marg_res.to_csv(args.out+'.marg.out',sep='\t',na_rep='NA',
                                float_format='%.5g')
        # chi_res.to_csv(args.out+'.chis.out',sep='\t',na_rep='NA',
        #                float_format='%.5g')

    print('Analysis finished at {T}'.format(T=time.ctime()))
    time_elapsed = round(time.time()-start_time,2)
    print('Total time elapsed: {T}'.format(T=time_elapsed))

if __name__ == '__main__':
    main(sys.argv)
