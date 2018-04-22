#!/usr/bin/env python
from typing import List, Any

from tino_cta.helper_functions import *
from astropy import units as u

from sys import exit, path
from glob import glob
from collections import namedtuple

# PyTables
import tables as tb

# ctapipe
from ctapipe.utils import linalg
from ctapipe.utils.CutFlow import CutFlow

from ctapipe.instrument import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters_4 as hillas_parameters
from ctapipe.coordinates.coordinate_transformations import alt_to_theta, az_to_phi
from ctapipe.reco.HillasReconstructor import HillasReconstructor
from ctapipe.reco.energy_regressor import *

# tino_cta
from tino_cta.ImageCleaning import ImageCleaner
from tino_cta.prepare_event import EventPreparer

if __name__ == "__main__":

    # your favourite units here
    energy_unit = u.TeV
    angle_unit = u.deg
    dist_unit = u.m

    parser = make_argparser()
    parser.add_argument('-o', '--outfile', type=str, required=True)

    parser.add_argument('--wave_dir', type=str, default=None,
                        help="directory where to find mr_filter. "
                             "if not set look in $PATH")
    parser.add_argument('--wave_temp_dir', type=str, default='/dev/shm/',
                        help="directory where mr_filter to store the temporary fits"
                             " files")
    parser.add_argument('--ascii_list', type=str, default=None,
                        help='ASCII list containing the name of the files to be used')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gamma', default=True, action='store_true',
                       help="do gammas (default)")
    group.add_argument('--proton', action='store_true',
                       help="do protons instead of gammas")
    group.add_argument('--electron', action='store_true',
                       help="do electrons instead of gammas")

    parser.add_argument('--estimate_energy', default=False, help='energy estimation')

    args = parser.parse_args()

    # Determine whether if energy is computed or not
    estimate_energy = args.estimate_energy
    print(args.estimate_energy)
    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
        filenamelist.sort()
    elif args.ascii_list is not None:
        filenamelist = [args.indir + '/' + line.rstrip('\n') for line in open(args.ascii_list)]
    elif args.proton:
        filenamelist = glob("{}/proton/*gz".format(args.indir))
        channel = "proton"
    elif args.electron:
        filenamelist = glob("{}/electron/*gz".format(args.indir))
        channel = "electron"
    elif args.gamma:
        filenamelist = glob("{}/gamma/*gz".format(args.indir))
        channel = "gamma"
    else:
        raise ValueError("don't know which input to use...")

    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)
    else:
        print("found {} files".format(len(filenamelist)))

    # keeping track of events and where they were rejected
    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    # takes care of image cleaning
    cleaner = ImageCleaner(mode=args.mode, cutflow=Imagecutflow,
                           wavelet_options=args.raw,
                           tmp_files_directory=args.wave_temp_dir,
                           skip_edge_events=True,
                           island_cleaning=True)

    # the class that does the shower reconstruction
    shower_reco = HillasReconstructor()

    preper = EventPreparer(
        cleaner=cleaner,
        hillas_parameters=hillas_parameters,
        shower_reco=shower_reco,
        event_cutflow=Eventcutflow,
        image_cutflow=Imagecutflow,
        # event/image cuts:
        allowed_cam_ids=[],
        min_ntel=2,
        min_charge=args.min_charge, min_pixel=3)

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    # wrapper for the scikit-learn regressor
    if estimate_energy == True:
        args_regressor = "./classifier_to_pickle/regressor_{mode}_{cam_id}_{regressor}.pkl"
        regressor = EnergyRegressor.load(
            args_regressor.format(**{
            "mode": args.mode,
            "wave_args": "mixed",
            "regressor": "RandomForestRegressor",
            "cam_id": "{cam_id}"}),
            cam_id_list=args.cam_ids)

        EnergyFeatures = namedtuple(
            "EnergyFeatures", (
            "impact_dist",
            "sum_signal_evt",
            "width",
            "length",
            "h_max",
            "local_distance")
        )

    class EventFeatures(tb.IsDescription):
        impact_dist = tb.Float32Col(dflt=1, pos=0)
        sum_signal_evt = tb.Float32Col(dflt=1, pos=1)
        max_signal_cam = tb.Float32Col(dflt=1, pos=2)
        sum_signal_cam = tb.Float32Col(dflt=1, pos=3)
        N_LST = tb.Int16Col(dflt=1, pos=4)
        N_MST = tb.Int16Col(dflt=1, pos=5)
        N_SST = tb.Int16Col(dflt=1, pos=6)
        width = tb.Float32Col(dflt=1, pos=7)
        length = tb.Float32Col(dflt=1, pos=8)
        skewness = tb.Float32Col(dflt=1, pos=9)
        kurtosis = tb.Float32Col(dflt=1, pos=10)
        h_max = tb.Float32Col(dflt=1, pos=11)
        err_est_pos = tb.Float32Col(dflt=1, pos=12)
        err_est_dir = tb.Float32Col(dflt=1, pos=13)
        MC_Energy = tb.FloatCol(dflt=1, pos=14)
        local_distance = tb.Float32Col(dflt=1, pos=15)
        n_pixel = tb.Int16Col(dflt=1, pos=16)
        n_cluster = tb.Int16Col(dflt=-1, pos=17)
        run_id = tb.Int16Col(dflt=1, pos=18)
        event_id = tb.Int16Col(dflt=1, pos=19)
        tel_id = tb.Int16Col(dflt=1, pos=20)
        xi = tb.Float32Col(dflt=np.nan, pos=21)
        reco_energy = tb.FloatCol(dflt=np.nan, pos=22)


    feature_outfile  = tb.open_file(args.outfile, mode="w")
    feature_table = {}
    feature_events = {}

    # mc_energy = []  # type: List[Any]
    # reco_energy = []

    # Full-array south
    #allowed_tels = set(prod3b_tel_ids("L+N+D"))
    # Subarray LSTs, south
    # allowed_tels = set(prod3b_tel_ids('subarray_LSTs', site='south'))

    # Subarray LSTs, North
    allowed_tels = set(prod3b_tel_ids('subarray_LSTs', site='north'))

    #for i, filename in enumerate(filenamelist[:50][:args.last]):
    for i, filename in enumerate(filenamelist[:args.last]):
        print("file: {} filename = {}".format(i, filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        # loop that cleans and parametrises the images and performs the reconstruction
        for (event, n_pixel_dict, hillas_dict, n_tels,
             tot_signal, max_signals, pos_fit, dir_fit, h_max,
             err_est_pos, err_est_dir) in preper.prepare_event(source):

            # the MC direction of origin of the simulated particle
            shower = event.mc
            shower_core = np.array([shower.core_x / u.m,
                                    shower.core_y / u.m]) * u.m
            shower_org = linalg.set_phi_theta(az_to_phi(shower.az),
                                              alt_to_theta(shower.alt))
                    
            # and how the reconstructed direction compares to that
            xi = linalg.angle(dir_fit, shower_org)
            
            n_faint = 0

            reco_energy = np.nan
            # Not optimal at all, two loop on tel!!!
            # For energy estimation
            if estimate_energy == True:
                reg_features_evt = {}
                for tel_id in hillas_dict.keys():
                    cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                    moments = hillas_dict[tel_id]
                    tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m
                    impact_dist = linalg.length(tel_pos - pos_fit)

                    reg_features_tel = EnergyFeatures(
                        impact_dist=impact_dist / u.m,
                        sum_signal_evt=tot_signal,
                        width=moments.width / u.m,
                        length=moments.length / u.m,
                        h_max=h_max / u.m,
                        local_distance=moments.r / moments.r.unit
                    )

                    try:
                        reg_features_evt[cam_id] += [reg_features_tel]
                    except KeyError:
                        reg_features_evt[cam_id] = [reg_features_tel]

                if reg_features_evt:
                    predict_energy = regressor.predict_by_event([reg_features_evt])["mean"][0]
                    reco_energy = predict_energy.to(energy_unit).value
                else:
                    reco_energy = np.nan

            for tel_id in hillas_dict.keys():
                cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                
                if cam_id not in feature_events:
                    feature_table[cam_id] = feature_outfile.create_table(
                        '/', '_'.join(["feature_events", cam_id]), EventFeatures)
                    feature_events[cam_id] = feature_table[cam_id].row

                moments = hillas_dict[tel_id]
                tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m
                impact_dist = linalg.length(tel_pos - pos_fit)

                feature_events[cam_id]["impact_dist"] = impact_dist / dist_unit
                feature_events[cam_id]["sum_signal_evt"] = tot_signal
                feature_events[cam_id]["max_signal_cam"] = max_signals[tel_id]
                feature_events[cam_id]["sum_signal_cam"] = moments.size
                feature_events[cam_id]["N_LST"] = n_tels["LST"]
                feature_events[cam_id]["N_MST"] = n_tels["MST"]
                feature_events[cam_id]["N_SST"] = n_tels["SST"]
                feature_events[cam_id]["width"] = moments.width / dist_unit
                feature_events[cam_id]["length"] = moments.length / dist_unit
                feature_events[cam_id]["skewness"] = moments.skewness
                feature_events[cam_id]["kurtosis"] = moments.kurtosis
                feature_events[cam_id]["h_max"] = h_max / dist_unit
                feature_events[cam_id]["err_est_pos"] = err_est_pos / dist_unit
                feature_events[cam_id]["err_est_dir"] = err_est_dir / angle_unit
                feature_events[cam_id]["MC_Energy"] = event.mc.energy / energy_unit
                feature_events[cam_id]["local_distance"] = moments.r / moments.r.unit
                feature_events[cam_id]["n_pixel"] = n_pixel_dict[tel_id]
                feature_events[cam_id]["run_id"] = event.r0.run_id
                feature_events[cam_id]["event_id"] = event.r0.event_id
                feature_events[cam_id]["tel_id"] = tel_id
                feature_events[cam_id]["xi"] = xi / angle_unit
                feature_events[cam_id]["reco_energy"] = reco_energy
                feature_events[cam_id].append()


            if signal_handler.stop:
                break
        if signal_handler.stop:
            break
    # make sure that all the events are properly stored
    for table in feature_table.values():
        table.flush()

    Imagecutflow()
    Eventcutflow()
