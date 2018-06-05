import numpy as np
from astropy import units as u
import warnings
from traitlets.config import Config
from collections import namedtuple, OrderedDict

# CTAPIPE utilities
from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas
from ctapipe.utils.linalg import rotation_matrix_2d
from ctapipe.utils.CutFlow import CutFlow
from ctapipe.coordinates.coordinate_transformations import (
    az_to_phi, alt_to_theta, transform_pixel_position)

# Tino's utilities
from tino_cta.ImageCleaning import ImageCleaner, EdgeEvent

# PiWy utilities
from pywicta.io import geometry_converter
from pywicta.io.images import simtel_event_to_images
from pywi.processing.filtering import pixel_clusters

# monkey patch the camera calibrator to do NO integration correction
import ctapipe
def null_integration_correction_func(n_chan, pulse_shape, refstep, time_slice,
                                     window_width, window_shift):
    return np.ones(n_chan)
# apply the patch
ctapipe.calib.camera.dl1.integration_correction = null_integration_correction_func


PreparedEvent = namedtuple("PreparedEvent",
                           ["event", "n_pixel_dict", "hillas_dict", "n_tels",
                            "tot_signal", "max_signals",
                            "pos_fit", "dir_fit", "h_max",
                            "err_est_pos", "err_est_dir", "n_cluster_dict"
                            ])

def stub(event):
    return PreparedEvent(event=event, n_pixel_dict=None, hillas_dict=None, n_tels=None,
                         tot_signal=None, max_signals=None,
                         pos_fit=None, dir_fit=None, h_max=None,
                         err_est_pos=None, err_est_dir=None, n_cluster_dict=None)


tel_phi = {}
tel_theta = {}
tel_orientation = (tel_phi, tel_theta)

# use this in the selection of the gain channels
np_true_false = np.array([[True], [False]])


def raise_error(message):
    raise ValueError(message)


class EventPreparer():

    # for gain channel selection
    pe_thresh = {
        "ASTRICam": 14,
        "LSTCam": 100,
        "NectarCam": 190}

    def __init__(self, calib=None, cleaner=None, hillas_parameters=None,
                 shower_reco=None, event_cutflow=None, image_cutflow=None,
                 # event/image cuts:
                 allowed_cam_ids=None, min_ntel=2, min_charge=50, min_pixel=3):

        # configuration for the camera calibrator
        # modifies the integration window to be more like in MARS
        # JLK, only for LST!!!!
        # OPtion for integration correction is done above
        cfg = Config()
        cfg["ChargeExtractorFactory"]["extractor"] = 'LocalPeakIntegrator'
        cfg["ChargeExtractorFactory"]["window_width"] = 5
        cfg["ChargeExtractorFactory"]["window_shift"] = 2

        self.calib = calib or CameraCalibrator(config=cfg, tool=None)
        self.cleaner = cleaner or ImageCleaner(mode=None)
        self.hillas_parameters = hillas_parameters or hillas.hillas_parameters
        self.shower_reco = shower_reco or \
            raise_error("need to provide a shower reconstructor....")

        # adding cutflows and cuts for events and images
        self.event_cutflow = event_cutflow or CutFlow("EventCutFlow")
        self.image_cutflow = image_cutflow or CutFlow("ImageCutFlow")

        self.event_cutflow.set_cuts(OrderedDict([
            ("noCuts", None),
            ("min2Tels trig", lambda x: x < min_ntel),
            ("min2Tels reco", lambda x: x < min_ntel),
            ("position nan", lambda x: np.isnan(x.value).any()),
            ("direction nan", lambda x: np.isnan(x.value).any())
        ]))

        ### JLK note, nominal distance cut should be done here, not in image cleaning...
        self.image_cutflow.set_cuts(OrderedDict([
            ("noCuts", None),
            ("min pixel", lambda s: np.count_nonzero(s) < min_pixel),
            ("min charge", lambda x: x < min_charge),
            ("poor moments", lambda m: m.width <= 0 or m.length <= 0),
            ("bad ellipticity", lambda m: (m.width/m.length) < 0.1 or (m.width/m.length) > 0.6)
        ]))

    @classmethod
    def pick_gain_channel(cls, pmt_signal, cam_id):
        '''the PMTs on some (most?) cameras have 2 gain channels. select one
        according to a threshold. ultimately, this will be done IN the
        camera/telescope itself but until then, do it here
        '''

        if pmt_signal.shape[0] > 1:
            pmt_signal = np.squeeze(pmt_signal)
            pick = (cls.pe_thresh[cam_id] <
                    pmt_signal).any(axis=0) != np_true_false
            pmt_signal = pmt_signal.T[pick.T]
        else:
            pmt_signal = np.squeeze(pmt_signal)
        return pmt_signal

    def prepare_event(self, source, return_stub=False):

        for event in source:

            self.event_cutflow.count("noCuts")

            if self.event_cutflow.cut("min2Tels trig", len(event.dl0.tels_with_data)):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            # calibrate the event
            self.calib.calibrate(event)

            # telescope loop
            tot_signal = 0
            max_signals = {}
            n_pixel_dict = {}
            hillas_dict = {}
            n_tels = {"tot": len(event.dl0.tels_with_data),
                      "LST": 0, "MST": 0, "SST": 0}
            n_cluster_dict = {}
            for tel_id in event.dl0.tels_with_data:
                self.image_cutflow.count("noCuts")
                
                camera = event.inst.subarray.tel[tel_id].camera

                # can this be improved?
                if tel_id not in tel_phi:
                    tel_phi[tel_id] = az_to_phi(event.mc.tel[tel_id].azimuth_raw * u.rad)
                    tel_theta[tel_id] = \
                        alt_to_theta(event.mc.tel[tel_id].altitude_raw * u.rad)

                    # the orientation of the camera (i.e. the pixel positions) needs to
                    # be corrected
                    camera.pix_x, camera.pix_y = \
                        transform_pixel_position(camera.pix_x, camera.pix_y)

                # count the current telescope according to its size
                tel_type = event.inst.subarray.tel[tel_id].optics.tel_type
                # JLK, N telescopes are not really interesting for discrimination, too much fluctuations
                # n_tels[tel_type] += 1

                # the camera image as a 1D array and stuff needed for calibration
                #pmt_signal = event.dl1.tel[tel_id].image
                #pmt_signal = self.pick_gain_channel(pmt_signal, camera.cam_id)
                # Choose gain according to pywicta's procedure
                image_1d = simtel_event_to_images(event=event, tel_id=tel_id, ctapipe_format=True)
                pmt_signal = image_1d.input_image  # calibrated image

                # clean the image
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pmt_signal, new_geom = \
                            self.cleaner.clean(pmt_signal.copy(), camera)

                    if self.image_cutflow.cut("min pixel", pmt_signal) or \
                       self.image_cutflow.cut("min charge", np.sum(pmt_signal)):
                        continue

                except FileNotFoundError as e:
                    print(e)
                    continue
                except EdgeEvent:
                    continue

                # For cluster counts
                image_2d = geometry_converter.image_1d_to_2d(pmt_signal, camera.cam_id)
                n_cluster_dict[tel_id] = pixel_clusters.number_of_pixels_clusters(
                    array=image_2d,
                    threshold=0
                )
                # print('==> Prepare, #pix={}'.format(n_cluster_dict[tel_id]))

                # could this go into `hillas_parameters` ...?
                max_signals[tel_id] = np.max(pmt_signal)

                # do the hillas reconstruction of the images
                # QUESTION should this change in numpy behaviour be done here
                # or within `hillas_parameters` itself?
                with np.errstate(invalid='raise', divide='raise'):
                    try:
                        moments = self.hillas_parameters(new_geom, pmt_signal)

                    # if width and/or length are zero (e.g. when there is only only one
                    # pixel or when all  pixel are exactly in one row), the
                    # parametrisation won't be very useful: skip
                        if self.image_cutflow.cut("poor moments", moments):
                            continue

                        if self.image_cutflow.cut("bad ellipticity", moments):
                            continue

                    except (FloatingPointError, hillas.HillasParameterizationError):
                        continue

                n_tels[tel_type] += 1
                hillas_dict[tel_id] = moments
                n_pixel_dict[tel_id] = len(np.where(pmt_signal>0)[0])
                tot_signal += moments.size
                image_2d = geometry_converter.image_1d_to_2d(pmt_signal, camera.cam_id)


            n_tels["reco"] = len(hillas_dict)
            if self.event_cutflow.cut("min2Tels reco", n_tels["reco"]):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # telescope loop done, now do the core fit
                    self.shower_reco.get_great_circles(
                        hillas_dict, event.inst.subarray, tel_phi, tel_theta)
                    pos_fit, err_est_pos = self.shower_reco.fit_core_crosses()
                    dir_fit, err_est_dir = self.shower_reco.fit_origin_crosses()
                    h_max = self.shower_reco.fit_h_max(
                        hillas_dict, event.inst.subarray, tel_phi, tel_theta
                    )
            except Exception as e:
                print("exception in reconstruction:", e)
                if return_stub:
                    yield stub(event)
                else:
                    continue

            if self.event_cutflow.cut("position nan", pos_fit) or \
               self.event_cutflow.cut("direction nan", dir_fit):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            yield PreparedEvent(event=event, n_pixel_dict=n_pixel_dict, hillas_dict=hillas_dict,
                                n_tels=n_tels, tot_signal=tot_signal, max_signals=max_signals,
                                pos_fit=pos_fit, dir_fit=dir_fit, h_max=h_max,
                                err_est_pos=err_est_pos, err_est_dir=err_est_dir, n_cluster_dict=n_cluster_dict
                                )

