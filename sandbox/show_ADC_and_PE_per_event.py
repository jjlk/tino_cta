import numpy as np
from glob import glob
import argparse
from ctapipe.io.hessio import hessio_event_source
from astropy import units as u
import sys

from itertools import chain

from numpy import ceil
from matplotlib import pyplot as plt

from ctapipe.reco.hillas import hillas_parameters,hillas_parameters_2
from ctapipe.reco.cleaning import tailcuts_clean, dilate
from ctapipe import visualization, io

def get_input():
    print("============================================")
    print("n or [enter]    - go to Next event")
    print("d               - Display the event")
    print("p               - Print all event data")
    print("i               - event Info")
    print("q               - Quit")
    return input("Choice: ")

fig = plt.figure(figsize=(12, 8))
def display_event(event, calibrate = 0, max_tel = 4, cleaning=None):
    """an extremely inefficient display. It creates new instances of
    CameraDisplay for every event and every camera, and also new axes
    for each event. It's hacked, but it works
    """
    print("Displaying... please wait (this is an inefficient implementation)")
    global fig
    ntels = min(max_tel, len(event.dl0.tels_with_data))
    fig.clear()

    plt.suptitle("EVENT {}".format(event.dl0.event_id))

    disps = []

    for ii, tel_id in enumerate(event.dl0.tels_with_data):
        if ii >= max_tel: break
        print("\t draw cam {}...".format(tel_id))
        nn = int(ceil((ntels)**.5))
        ax = plt.subplot(nn, 2*nn, 2*(ii+1)-1)

        x, y = event.meta.pixel_pos[tel_id]
        geom = io.CameraGeometry.guess(x, y, event.meta.optical_foclen[tel_id])
        disp = visualization.CameraDisplay(geom, ax=ax,
                                           title="CT{0} DetectorResponse".format(tel_id))
        
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.cmap = plt.cm.hot
        chan = 0
        signals = event.dl0.tel[tel_id].adc_sums[chan].astype(float)[:]
        if calibrate:
            signals = apply_mc_calibration_ASTRI(event.dl0.tel[tel_id].adc_sums, tel_id)
        if cleaning == 'tailcut':
            mask = tailcuts_clean(geom, signals, 1,picture_thresh=10.,boundary_thresh=8.)
            dilate(geom, mask)
            signals[mask==False] = 0
            
        moments = hillas_parameters_2(geom.pix_x,
                                    geom.pix_y,
                                    signals)

        disp.image = signals
        disp.overlay_moments(moments, color='seagreen', linewidth=3)
        disp.set_limits_percent(95)
        disp.add_colorbar()
        disps.append(disp)
        
        
        ax = plt.subplot(nn, 2*nn, 2*(ii+1))

        geom = io.CameraGeometry.guess(x, y, event.meta.optical_foclen[tel_id])
        disp = visualization.CameraDisplay(geom, ax=ax,
                                           title="CT{0} PhotoElectrons".format(tel_id))
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.cmap = plt.cm.hot
        chan = 0


        #print (event.mc.tel[tel_id].photo_electrons)
        for jj in range(len(event.mc.tel[tel_id].photo_electrons)):
            event.dl0.tel[tel_id].adc_sums[chan][jj] = event.mc.tel[tel_id].photo_electrons[jj]
        signals2 = event.dl0.tel[tel_id].adc_sums[chan].astype(float)
        moments2 = hillas_parameters_2(geom.pix_x,
                                    geom.pix_y,
                                    signals2)


        disp.image = signals2
        #disp.overlay_moments(moments2, color='seagreen', linewidth=3)
        disp.set_limits_percent(95)
        disp.add_colorbar()
        disps.append(disp)
        

    return disps

import pyhessio
def get_mc_calibration_coeffs(tel_id):
    """
    Get the calibration coefficients from the MC data file to the
    data.  This is ahack (until we have a real data structure for the
    calibrated data), it should move into `ctapipe.io.hessio_event_source`.

    returns
    -------
    (peds,gains) : arrays of the pedestal and pe/dc ratios.
    """
    peds = pyhessio.get_pedestal(tel_id)[0]
    gains = pyhessio.get_calibration(tel_id)[0]
    return peds, gains


def apply_mc_calibration(adcs, tel_id):
    """
    apply basic calibration
    """
    peds, gains = get_mc_calibration_coeffs(tel_id)

    if adcs.ndim > 1:  # if it's per-sample need to correct the peds
        return ((adcs - peds[:, np.newaxis] / adcs.shape[1]) *
                gains[:, np.newaxis])

    return (adcs - peds) * gains

def apply_mc_calibration_ASTRI(adcs, tel_id):
    """
    apply basic calibration
    """
    
    peds0 = pyhessio.get_pedestal(tel_id)[0]
    peds1 = pyhessio.get_pedestal(tel_id)[1]
    gains0 = pyhessio.get_calibration(tel_id)[0]
    gains1 = pyhessio.get_calibration(tel_id)[1]
    
    calibrated = [ (adc0-971)*gain0 if adc0 < 3500 else (adc1-961)*gain1 for adc0, adc1, gain0, gain1 in zip(adcs[0], adcs[1], gains0,gains1) ]
    return np.array(calibrated)


if __name__ == '__main__':
    


    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-t','--tel', type=int)
    parser.add_argument('-r', '--runnr',   type=str, default="*")
    parser.add_argument('-i', '--indir',   type=str, 
                        default="/local/home/tmichael/Data/cta/ASTRI9")
    parser.add_argument('-m', '--max-events', type=int, default=10000)
    parser.add_argument('-n', '--max-ntels',  type=int, default=4)
    parser.add_argument('-w', '--write', action='store_true',
                        help='write images to files')
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-s', '--show-samples', action='store_true',
                        help='show time-variablity, one frame at a time')
    parser.add_argument('-c','--calibrate', action='store_true',
                        help='apply calibration coeffs from MC')
    parser.add_argument('--clean', type=str,
                        help='apply given cleaning method')
    args = parser.parse_args()

    filenamelist_gamma  = glob( "{}/gamma/run{}.*gz".format(args.indir,args.runnr ))
    filenamelist_proton = glob( "{}/proton/run{}.*gz".format(args.indir,args.runnr ))
    
    print(  "{}/gamma/run{}.*gz".format(args.indir,args.runnr ))
    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()
    if len(filenamelist_proton) == 0:
        print("no protons found")
        exit()

    for filename in chain(sorted(filenamelist_gamma)[:], sorted(filenamelist_proton)[:]):
        print("filename = {}".format(filename))
        
        source = hessio_event_source(filename,
                                    allowed_tels=[6],#range(10),
                                    max_events=args.max_events)

        for event in source:

            print('Scanning input file... count = {}'.format(event.count))
            print(event.dl0.tels_with_data)
            if args.tel and args.tel not in event.dl0.tels_with_data: continue

                    
            while True:
                response = get_input()
                print()
                if response.startswith("d"):
                    disps = display_event(event,max_tel=args.max_ntels,
                                          calibrate=args.calibrate,
                                          cleaning=args.clean)
                    plt.pause(0.1)
                elif response.startswith("p"):
                    print("--event-------------------")
                    print(event)
                    print("--event.dl0---------------")
                    print(event.dl0)
                    print("--event.dl0.tel-----------")
                    for teldata in event.dl0.tel.values():
                        print(teldata)
                elif response == "" or response.startswith("n"):
                    break
                elif response.startswith('i'):
                    for tel_id in sorted(event.dl0.tel):
                        for chan in event.dl0.tel[tel_id].adc_samples:
                            npix = len(event.meta.pixel_pos[tel_id][0])
                            print("CT{:4d} ch{} pixels:{} samples:{}"
                                .format(tel_id, chan, npix,
                                        event.dl0.tel[tel_id].
                                        adc_samples[chan].shape[1]))

                elif response.startswith('q'):
                    sys.exit()
                else:
                    sys.exit()

            if response.startswith('q'):
                sys.exit()
