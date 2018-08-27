import numpy as np
import astropy.units as u
from scipy import ndimage

from ctapipe.utils.CutFlow import CutFlow

from ctapipe.image.cleaning import tailcuts_clean, dilate
from ctapipe.image.geometry_converter import (convert_geometry_hex1d_to_rect2d,
                                              convert_geometry_rect2d_back_to_hexe1d,
                                              astri_to_2d_array, array_2d_to_astri,
                                              chec_to_2d_array, array_2d_to_chec)

try:
    # OLD OLD
    #from pywicta.denoising.wavelets_mrfilter import WaveletTransform
    from pywicta.denoising.wavelets_mrtransform import WaveletTransform
    from pywicta.denoising import cdf
    from pywicta.denoising.inverse_transform_sampling import \
        EmpiricalDistribution as EmpDist
    from pywicta.io import geometry_converter
    from pywi.processing.filtering.pixel_clusters import filter_pixels_clusters
    from pywi.processing.filtering import pixel_clusters
except ImportError as e:
    print("Jeremie's 'pywicta' package could not be imported")
    print("wavelet cleaning will not work;"
          " better use the '--tail' command line flag")
    print(e)


class UnknownMode(ValueError):
    pass


class EdgeEvent(RuntimeError):
    pass


class MissingImplementation(KeyError):
    pass


# def kill_isolpix(array, neighbours=None, threshold=.2):
#     """
#     Return array with isolated islands removed.
#     Only keeping the biggest islands (largest surface).
#
#     Parameters
#     ----------
#     array : 2D array
#         the input image you want to keep the "biggest" patch of
#     neighbours : 2D array, optional (default: None)
#         a mask defining what is considered a neighbour
#     threshold : float, optional (default: 0.2)
#         ignores pixel with entries below this value
#
#     Returns
#     -------
#     filtered_array : 2D array
#         image with just the largest island remaining
#     """
#
#     filtered_array = np.copy(array)
#
#     filtered_array[filtered_array < threshold] = 0
#     mask = filtered_array > 0
#
#     label_im, nb_labels = ndimage.label(mask, neighbours)
#
#     sums = ndimage.sum(filtered_array, label_im, range(nb_labels + 1))
#     mask_sum = sums < np.max(sums)
#     remove_pixel = mask_sum[label_im]
#
#     filtered_array[remove_pixel] = 0
#
#     return filtered_array


def get_edge_pixels(camera, rows=1, n_neigh=None):
    """collects a list of pixel IDs that are considered to be "the edge" of the image.

    Parameters
    ----------
    camera : ctapipe CameraGeometry object
        the camera geometry object
    rows : integer, optional (default: 1)
        width of the edge in pixel-rows
    n_neigh : integer, optional (default: None)
        number of nominal number of neighbours for each pixels
        gets set to 6 for hexagonal and 4 for rectangular pixels
        ... you probably don't need to worry about this ...

    Returns
    -------
        python set of pixel IDs to consider "the edge" of the image
        edge_pixels : set

    Note
    ----
    `edge_pixels` is stored as a member of `camera`, since the same camera geometry will
    show up many times
    """
    if not hasattr(camera, "edge_pixels"):
        # the first row consists of all pixels that have less than the nominal number of
        # neighbours -- which is 6 for hexagonal pixels and 4 for rectangular ones
        n_neigh = n_neigh or (6 if "hex" in camera.pix_type else 4)
        edge_pixels = set(id for id, neigh in zip(camera.pix_id, camera.neighbors)
                          if len(neigh) < n_neigh)

        # add more rows by joining `edge_pixels` with the sets of the neighbours of all
        # pixels in `edge_pixels`
        for i in range(rows - 1):
            for j in list(edge_pixels):
                edge_pixels |= set(camera.neighbors[j])

        camera.edge_pixels = np.array(list(edge_pixels), dtype=int)
    return camera.edge_pixels


def reject_edge_event(img, geom, rel_thresh=5., abs_thresh=None, rows=1):
    """determines whether any pixel at the edge (defined by `get_edge_pixels`) has a
    too high signal. The threshold is given either by `abs_thresh` or the highest signal
    divided by `rel_thresh`.

    Parameters
    ----------
    img : ndarray
        the camera image
    geom : ctapipe CameraGeometry object
        the camera geometry object
    rel_thresh : float, optional (default: 5.)
        divide the maximum signal by this number to get the threshold for a "too high
        signal at the edge"
    abs_thresh : float, optional (default: None)
        absolute value to consider a "too high signal at the edge"
        if given, overrides `rel_thresh` method

    Returns
    -------
    reject : bool
        whether or not any of the edge pixels has a signal higher than the threshold
    """
    edge_thresh = abs_thresh or (np.max(img) / rel_thresh)
    edge_pixels = get_edge_pixels(geom, rows=rows)
    return (img[edge_pixels] > edge_thresh).any()


def reject_event_radius(img, geom, rel_radius=.8, which="inner"):
    """Rejects an image depending on the shower core's distance to the camera centre.

    Parameters
    ----------
    img : ndarray
        the camera image
    geom : ctapipe CameraGeometry object
        the camera geometry object
    rel_radius : float 0 < x < 1, (default: 0.8)
        fraction of the camera extension within which the shower core shall lie
    which : string ["inner" | "outer"], (default: "inner")
        sets the closest of furthest edge pixel as the radius of the camera image

    Returns
    -------
    True if shower core is outside the allowed range (so: yes, reject)
    False if shower core is within the allowed range (so: no, don't reject)

    Note
    ----
    Assumes image centre is at geom.pix_x = geom.pix_y = 0.

    """

    edge_pixels = get_edge_pixels(geom, rows=1)
    if which is "inner":
        r_edge_squared = np.min((geom.pix_x**2 + geom.pix_y**2)[edge_pixels])
    elif wich is "outer":
        r_edge_squared = np.max((geom.pix_x**2 + geom.pix_y**2)[edge_pixels])
    else:
        raise KeyError("'which' can only be 'inner' or 'outer', got:",
                       {which})

    try:
        cen_x = np.average(geom.pix_x, weights=img)
        cen_y = np.average(geom.pix_y, weights=img)
    except ZeroDivisionError:
        cen_x, cen_y = 0, 0

    # LST hack, to be consistent with Abelardo and Jeremie
    r_cam_max = 1.0904 * u.m

    #return cen_x**2 + cen_y**2 > r_edge_squared * rel_radius**2
    return cen_x ** 2 + cen_y ** 2 > (r_cam_max * rel_radius) ** 2


class ImageCleaner:

    hex_neighbours_1ring = np.array([[1, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 1]])
    hex_neighbours_2ring = np.array([[1, 1, 1, 0, 0],
                                     [1, 1, 1, 1, 0],
                                     [1, 1, 1, 1, 1],
                                     [0, 1, 1, 1, 1],
                                     [0, 0, 1, 1, 1]])

    geom_1d_to_2d = {"ASTRICam": astri_to_2d_array,
                     "CHEC": chec_to_2d_array,
                     "hex": convert_geometry_hex1d_to_rect2d}

    geom_2d_to_1d = {"ASTRICam": array_2d_to_astri,
                     "CHEC": array_2d_to_chec,
                     "hex": convert_geometry_rect2d_back_to_hexe1d}

    def __init__(self, mode="wave", dilate=False, island_cleaning=True,
                 skip_edge_events=False,  # JLK, now done in prepare event as a std cut
                 edge_width=1,
                 cutflow=CutFlow("ImageCleaner"),
                 wavelet_options=None,
                 tmp_files_directory='/dev/shm/', mrfilter_directory=None):
        self.mode = mode
        self.edge_width = edge_width
        self.cutflow = cutflow

        if skip_edge_events:
            self.reject_edge_event = reject_event_radius
        else:
            self.reject_edge_event = lambda *a, **b: False
        self.cutflow.add_cut("edge event", self.reject_edge_event)

        if mode in [None, "none", "None"]:
            self.clean = self.clean_none
        elif mode.startswith("wave"):
            self.clean = self.clean_wave
            # self.wavelet_cleaning = \
            #     lambda *arg, **kwargs: WaveletTransform().clean_image(
            #         *arg, **kwargs,
            #         #kill_isolated_pixels=island_cleaning,
            #         kill_isolated_pixels=False,  # JLK test
            #         tmp_files_directory=tmp_files_directory,
            #         mrfilter_directory=mrfilter_directory)
            # JLK, Hack on hack...
            self.wavelet_cleaning = \
                lambda *arg, **kwargs: WaveletTransform().clean_image(
                    *arg, **kwargs,
                    type_of_filtering='hard_filtering',
                    filter_thresholds=[3, 0.2],
                    last_scale_treatment='drop',
                    kill_isolated_pixels=True,
                    detect_only_positive_structures=False,
                    tmp_files_directory=tmp_files_directory,
                    clusters_threshold=0)
                    #mrfilter_directory=mrfilter_directory)
            # self.island_threshold = 1.5

            # command line parameters for the mr_filter call
            self.wavelet_options = \
                {"ASTRICam": wavelet_options or "-K -C1 -m3 -s2,2,3,3 -n4",
                 "DigiCam": wavelet_options or "-K -C1 -m3 -s6.274,2.629,7.755,0.076 -n4",
                 "FlashCam": wavelet_options or "-K -C1 -m3 -s4, 4, 5, 4 -n4",
                 "NectarCam": wavelet_options or
                    "-K -C1 -m3 -s13.013,2.549,6.559,1.412 -n4",
                 #"LSTCam": wavelet_options or "-K -C1 -m3 -s23.343,2.490,-2.856,-0.719 -n4",
                 "LSTCam": wavelet_options or "-f hard_filtering -t 3,0.2 -L drop --kill-isolated-pixels --cluster-threshold=0",
                 # WARNING: DUMMY VALUES
                 "CHEC": wavelet_options or "-K -C1 -m3 -s2,2,3,3 -n4"
                 }

            # WARNING WARNING WARNING
            # camera models for noise injection
            # JLK: Need to be updated if calib/signal integration is changed!!!!
            self.noise_model = \
                {"ASTRICam": EmpDist(cdf.ASTRI_CDF_FILE),
                 "DigiCam": EmpDist(cdf.DIGICAM_CDF_FILE),
                 "FlashCam": EmpDist(cdf.FLASHCAM_CDF_FILE),
                 "NectarCam": EmpDist(cdf.NECTARCAM_CDF_FILE),
                 "LSTCam": EmpDist(cdf.LSTCAM_CDF_FILE),
                 # WARNING: DUMMY FILE
                 "CHEC": EmpDist(cdf.DIGICAM_CDF_FILE)}

        elif mode.startswith("tail"):
            self.clean = self.clean_tail
            self.tail_thresholds = \
                {"ASTRICam": (5, 7),  # (5, 10)?
                 "FlashCam": (12, 15),
                 #"LSTCam": (5, 10),  # ?? (3, 6) for Abelardo...
                 "LSTCam": (3, 6),  # JLK, same as MARS analysis
                 # ASWG Zeuthen talk by Abelardo Moralejo:
                 "NectarCam": (3, 6),
                 # "FlashCam": (4, 8),  # there is some scaling missing?
                 "DigiCam": (3, 6),
                 "CHEC": (2, 4),
                 "SCTCam": (1.5, 3)}

            # self.island_threshold = 1.5
            self.dilate = dilate
        else:
            raise UnknownMode(
                'cleaning mode "{}" not found'.format(mode))

        if island_cleaning:
            # Add JD method
            #self.island_cleaning = kill_isolpix
            self.island_cleaning = filter_pixels_clusters
        else:
            # just a pass-through that does nothing
            # (saves an if-statement in every `clean` call)
            self.island_cleaning = lambda x, *args, **kw: x

    def clean_wave(self, img, cam_geom):
        if cam_geom.pix_type.startswith("hex"):
            new_img, new_geom = self.clean_wave_hex(img, cam_geom)
        elif cam_geom.pix_type.startswith("rect"):
            new_img, new_geom = self.clean_wave_rect(img, cam_geom)
        else:
            raise MissingImplementation("wavelet cleaning not yet implemented"
                                        " for geometry {}".format(cam_geom.cam_id))

        if self.cutflow.cut("edge event",
                            img=new_img, geom=new_geom):

            raise EdgeEvent

        return new_img, new_geom

    def clean_wave_rect(self, img, cam_geom):
        try:
            array2d_img = self.geom_1d_to_2d[cam_geom.cam_id](img)
        except KeyError:
            raise MissingImplementation("wavelet cleaning not yet implemented"
                                        " for geometry {}".format(cam_geom.cam_id))

        cleaned_img = self.wavelet_cleaning(
            array2d_img,
            raw_option_string=self.wavelet_options[cam_geom.cam_id],
            noise_distribution=self.noise_model[cam_geom.cam_id])

        self.cutflow.count("wavelet cleaning")

        # wavelet_transform still leaves some isolated pixels; remove them
        cleaned_img = self.island_cleaning(cleaned_img)

        new_img = self.geom_2d_to_1d[cam_geom.cam_id](cleaned_img)
        new_geom = cam_geom

        return new_img, new_geom

    def clean_wave_hex(self, img, cam_geom):
        #print("#JLK: HOHOHOHOHOHOH")

        # JLK: Use Pywi as convertor
        # rot_geom, rot_img = convert_geometry_hex1d_to_rect2d(
        #     cam_geom, img, cam_geom.cam_id)
        rot_img = geometry_converter.image_1d_to_2d(img, cam_geom.cam_id)

        # cleaned_img = self.wavelet_cleaning(rot_img,
        #                                     #raw_option_string=self.wavelet_options[cam_geom.cam_id],
        #                                     noise_distribution=self.noise_model[cam_geom.cam_id])

        wt_cleaner = WaveletTransform()
        cleaned_img = self.wavelet_cleaning(rot_img, noise_distribution=self.noise_model[cam_geom.cam_id])
        self.cutflow.count("wavelet cleaning")

        # cleaned_img = self.island_cleaning(cleaned_img,
        #                                    neighbours=self.hex_neighbours_1ring,
        #                                    threshold=self.island_threshold)
        cleaned_img = self.island_cleaning(cleaned_img)

        # JLK: Use Pywi as convertor
        # unrot_geom, unrot_img = convert_geometry_rect2d_back_to_hexe1d(
        #     rot_geom, cleaned_img, cam_geom.cam_id)

        unrot_img = geometry_converter.image_2d_to_1d(cleaned_img, cam_geom.cam_id)

        #return unrot_img, unrot_geom
        return unrot_img, cam_geom

    def clean_tail(self, img, cam_geom):
        mask = tailcuts_clean(
            cam_geom, img,
            picture_thresh=self.tail_thresholds[cam_geom.cam_id][1],
            boundary_thresh=self.tail_thresholds[cam_geom.cam_id][0],
            keep_isolated_pixels=False,
            min_number_picture_neighbors=2)
        if self.dilate:
            dilate(cam_geom, mask)
        img[~mask] = 0

        self.cutflow.count("tailcut cleaning")

        # Check for AstriCam and CHEC
        if "ASTRI" in cam_geom.cam_id:
            # turn into 2d to apply island cleaning
            img = astri_to_2d_array(img)

            # if set, remove all signal patches but the biggest one
            new_img = self.island_cleaning(img)

            # turn back into 1d array
            new_img = array_2d_to_astri(new_img)
            new_geom = cam_geom

        else:
            # turn into 2d to apply island cleaning
            # rot_geom, rot_img = convert_geometry_hex1d_to_rect2d(
            #     cam_geom, img, cam_geom.cam_id)

            rot_img = geometry_converter.image_1d_to_2d(img, cam_geom.cam_id)

            # if set, remove all signal patches but the biggest one
            # JLK we
            #cleaned_img = self.island_cleaning(rot_img, self.hex_neighbours_1ring)
            cleaned_img = self.island_cleaning(rot_img)

            # n_cluster = pixel_clusters.number_of_pixels_clusters(
            #     array=cleaned_img,
            #     threshold=0
            # )
            # print('JLK, #pix={}'.format(n_cluster))

            # turn back into 1d array
            # unrot_geom, unrot_img = convert_geometry_rect2d_back_to_hexe1d(
            #     rot_geom, cleaned_img, cam_geom.cam_id)

            unrot_img = geometry_converter.image_2d_to_1d(cleaned_img, cam_geom.cam_id)

            new_img = unrot_img
            #new_geom = unrot_geom
            new_geom = cam_geom

        # JLK, do not need that anymore, done in prepare_event
        # if self.cutflow.cut("edge event",
        #                     img=new_img, geom=new_geom):
        #     raise EdgeEvent

        return new_img, new_geom

    def clean_none(self, img, cam_geom):
        return img, cam_geom
