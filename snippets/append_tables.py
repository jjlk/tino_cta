#!/usr/bin/env python

from os.path import expandvars
import argparse

import glob

# PyTables
try:
    import tables as tb
except ImportError:
    print("no pytables installed?")

# pandas data frames
try:
    import pandas as pd
except ImportError:
    print("no pandas installed?")


def merge_list_of_pytables(filename_list, destination, table_name):
    pyt_table = None
    outfile = tb.open_file(destination, mode="w")
    for i, filename in enumerate(sorted(filename_list)):
        print(filename)

        pyt_infile = tb.open_file(filename, mode='r')

        if i == 0:
            pyt_table = pyt_infile.copy_node(
                where='/', name=table_name, newparent=outfile.root)

        else:
            #pyt_table_t = pyt_infile.root.reco_events
            # JLK hacked...
            # pyt_table_t = pyt_infile.root.feature_events_LSTCam
            pyt_table_t = pyt_infile.get_node('/' + table_name)
            pyt_table_t.append_where(dstTable=pyt_table)

    print("merged {} files".format(len(filename_list)))
    return pyt_table


def merge_list_of_pandas(filename_list, destination, table_name):
    store = pd.HDFStore(destination)
    for i, filename in enumerate(sorted(filename_list)):
        s = pd.HDFStore(filename)
        df = pd.read_hdf(filename, table_name)
        if i == 0:
            store.put(table_name, df, format='table', data_columns=True)
        else:
            store.append(key=table_name, value=df, format='table')
    return store[table_name]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--indir', type=str, default="./")
    parser.add_argument('--infiles_base', type=str, default="classified_events")
    parser.add_argument('--auto', action='store_true', dest='auto', default=False)
    parser.add_argument('-o', '--outfile', type=str)
    parser.add_argument('--table_name', type=str, default='reco_events')
    args = parser.parse_args()

    print('DEBUG> infiles_base={}'.format(args.infiles_base))
    print('DEBUG> indir={}'.format(args.indir))
    print('DEBUG> outfile={}'.format(args.outfile))
    print('DEBUG> auto={}'.format(args.auto))

    if args.auto:
        for channel in ["gamma", "proton"]:
            for mode in ["wave", "tail"]:
                filename = "{}/{}/{}_{}_{}_*.h5".format(
                    args.indir, mode,
                    args.infiles_base,
                    channel, mode)
                merge_list_of_pandas(glob.glob(filename),
                                     filename.replace("_*", ""))
    else:
        input_template = "{}/{}*.h5".format(args.indir, args.infiles_base)
        print("input_template:", input_template)

        filename_list = glob.glob(input_template)
        print("filename_list:", filename_list)

        merge_list_of_pytables(filename_list, args.outfile, args.table_name)
