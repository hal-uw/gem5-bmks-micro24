#!/usr/bin/env python

import argparse
import csv
import sys
import numpy as np

def fudge_rn(fudge_factor):
    return 2*fudge_factor*np.random.default_rng().random()-fudge_factor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=str, help='Filename '
                        'to output to')
    parser.add_argument('--num-queues', type=int, default=128,
                        help='Number of queues to gen test for')
    parser.add_argument('--max-kerns', type=int, default=5,
                        help='Max # of kernels per queue')
    parser.add_argument('--dl-scale', type=int, default=2,
                        help='Amount to scale raw deadline by')
    parser.add_argument('--dl-fudge', type=float, default=.2,
                        help='Fudge deadline by random number '
                        'in range of +- this')

    args = parser.parse_args()

    vals = np.random.default_rng().integers(1,args.max_kerns,
                                            args.num_queues,
                                            endpoint=True)

    with open(args.csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')

        for i in range(args.num_queues):
            times = np.random.default_rng().integers(1,4,vals[i], endpoint=True)
            rt_total = times.sum()
            rt_total *= args.dl_scale
            rt_total = rt_total + fudge_rn(args.dl_fudge)*rt_total
            write_arr = [rt_total, 0]+times.tolist()
            writer.writerow(write_arr)

if __name__ == '__main__':
    main()
