from extract_and_crop_simtel_images import crop_astri_image

from sys import exit, path
from os.path import expandvars
path.append(expandvars("$CTA_SOFT/tino_cta/"))
from modules.ImageCleaning import EdgeEventException, UnknownModeException
from helper_functions import *

old=False

''' old '''
from datapipe.denoising.wavelets_mrtransform import WaveletTransform as WaveletTransformOld
''' new '''
from datapipe.denoising.wavelets_mrfilter import WaveletTransform    as WaveletTransformNew


import numpy as np

from math import log10
from random import random
from itertools import chain

from astropy import units as u

from ctapipe.io import CameraGeometry
from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils import linalg
from ctapipe.utils.fitshistogram import Histogram

from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.image.cleaning import tailcuts_clean, dilate


from datapipe.utils.EfficiencyErrors import get_efficiency_errors

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm


class EventClassifier:

    def __init__(self,
                 class_list=['g', 'p'],
                 axisNames=["log(E / GeV)"],
                 ranges=[[2, 8]],
                 nbins=[6],):

        self.class_list = class_list
        self.axisNames = axisNames
        self.ranges = ranges
        self.nbins = nbins

        self.wrong = self.create_histogram_class_dict(class_list)
        self.total = self.create_histogram_class_dict(class_list)

        self.Features = self.create_empty_class_dict(class_list)
        self.MCEnergy = self.create_empty_class_dict(class_list)

        self.total_images    = 0
        self.selected_images = 0

    #def setup_geometry(self, h_telescopes, h_cameras, h_optics, phi=180.*u.deg, theta=20.*u.deg):
        #Ver = 'Feb2016'
        #TelVer = 'TelescopeTable_Version{}'.format(Ver)
        #CamVer = 'CameraTable_Version{}_TelID'.format(Ver)
        #OptVer = 'OpticsTable_Version{}_TelID'.format(Ver)

        #self.telescopes = h_telescopes[TelVer]
        #self.cameras    = lambda tel_id : h_cameras[CamVer+str(tel_id)]
        #self.optics     = lambda tel_id : h_optics [OptVer+str(tel_id)]

        #self.tel_phi   =  phi
        #self.tel_theta =  theta

        #self.tel_geom = {}
        #for tel_idx, tel_id in enumerate(self.telescopes['TelID']):
            #self.tel_geom[tel_id] = \
                #CameraGeometry.guess(self.cameras(tel_id)['PixX'].to(u.m),
                                     #self.cameras(tel_id)['PixY'].to(u.m),
                                     #self.telescopes['FL'][tel_idx] * u.m)

    def create_empty_class_dict(self, class_list):
        mydict = {}
        for cl in class_list:
            mydict[cl] = []
        return mydict

    def create_histogram_class_dict(self, class_list):
        mydict = {}
        for cl in class_list:
            mydict[cl] = Histogram(axisNames=self.axisNames,
                                   nbins=self.nbins, ranges=self.ranges)
        return mydict

    def equalise_nevents(self, NEvents):
        for cl in self.Features.keys():
            self.Features[cl] = self.Features[cl][:NEvents]
            self.MCEnergy[cl] = self.MCEnergy[cl][:NEvents]

    def learn(self, clf=None):
        trainFeatures   = []
        trainClasses    = []
        for cl in self.Features.keys():
            for ev in self.Features[cl]:
                trainFeatures += [tel[1:] for tel in ev]
                trainClasses  += [cl]*len(ev)

        if clf is None:
            clf = RandomForestClassifier(
                n_estimators=40, max_depth=None,
                min_samples_split=2, random_state=0)
        clf.fit(trainFeatures, trainClasses)
        self.clf = clf

    def save(self, path):
        from sklearn.externals import joblib
        joblib.dump(self.clf, path)

    def load(self, path):
        from sklearn.externals import joblib
        self.clf = joblib.load(path)

    def predict(self, ev):
        return self.clf.predict(ev)

    def show_importances(self, feature_labels=None):
        import matplotlib.pyplot as plt
        self.learn()
        importances = self.clf.feature_importances_
        bins = range(importances.shape[0])
        plt.figure()
        plt.title("Feature Importances")
        plt.bar(bins, importances,
                color='r', align='center')
        if feature_labels:
            plt.xticks(bins, feature_labels, rotation=17)

    def self_check(self, min_tel=3, agree_threshold=.75, clf=None,
                   split_size=None, verbose=True):
        import matplotlib.pyplot as plt

        right_ratios = self.create_empty_class_dict(self.class_list)
        proba_ratios = self.create_empty_class_dict(self.class_list)
        NTels        = self.create_empty_class_dict(self.class_list)

        start = 0
        NEvents = min(len(features)
                      for features in self.Features.values())

        if split_size is None:
            split_size = 10*max(NEvents//1000, 1)

        print("nevents:", NEvents)
        while start+split_size <= NEvents:
            trainFeatures = []
            trainClasses  = []
            '''
            training the classifier on all events but a chunck taken
            out at a certain position '''
            for cl in self.Features.keys():
                for ev in chain(self.Features[cl][:start],
                                self.Features[cl][start+split_size:]):
                    trainFeatures += [tel[1:] for tel in ev]
                    trainClasses  += [cl]*len(ev)

            if clf is None:
                clf = RandomForestClassifier(n_estimators=40, max_depth=None,
                                             min_samples_split=2, random_state=0)
            clf.fit(trainFeatures, trainClasses)

            '''
            test the training on the previously excluded chunck '''
            for cl in self.Features.keys():
                for ev, en in zip(
                        self.Features[cl][start:start+split_size],
                        self.MCEnergy[cl][start:start+split_size]
                                  ):

                    log_en = np.log10(en/u.GeV)

                    PredictTels = clf.predict([tel[:1]+tel[2:] for tel in ev])

                    predict_proba = clf.predict_proba([tel[:1]+tel[2:] for tel in ev])

                    proba_index = 0 if cl == "g" else 1
                    proba_ratios[cl].append(np.sum(np.sqrt(predict_proba), axis=0)[proba_index] /
                                            len(predict_proba))

                    NTels[cl].append(len(predict_proba))

                    '''
                    check if prediction was right '''
                    right_ratio = (len(PredictTels[PredictTels == cl]) /
                                   len(PredictTels))
                    right_ratios[cl].append(right_ratio)

                    '''
                    check if prediction returned gamma '''
                    gamma_ratio = (len(PredictTels[PredictTels == 'g']) /
                                   len(PredictTels))

                    '''
                    if sufficient telescopes agree, assume it's a gamma '''
                    if gamma_ratio > agree_threshold:
                        PredictClass = "g"
                    else:
                        PredictClass = "p"
                    if PredictClass != cl and len(ev) >= min_tel:

                        self.wrong[cl].fill([log_en])
                    self.total[cl].fill([log_en])

                if verbose and sum(self.total[cl].hist) > 0:
                    print("wrong {}: {} out of {} => {}".format(
                                    cl,
                                    sum(self.wrong[cl].hist),
                                    sum(self.total[cl].hist),
                                    sum(self.wrong[cl].hist) /
                                    sum(self.total[cl].hist) * 100*u.percent))

            start += split_size

            print()

        print()
        print("-"*30)
        print()
        y_eff         = self.create_empty_class_dict(self.class_list)
        y_eff_lerrors = self.create_empty_class_dict(self.class_list)
        y_eff_uerrors = self.create_empty_class_dict(self.class_list)

        try:
            from utils.EfficiencyErrors import get_efficiency_errors \
                as get_efficiency_errors
        except ImportError:
            pass

        for cl in self.Features.keys():
            if sum(self.total[cl].hist) > 0:
                print("wrong {}: {} out of {} => {}"
                      .format(cl, sum(self.wrong[cl].hist),
                                  sum(self.total[cl].hist),
                                  sum(self.wrong[cl].hist) /
                                  sum(self.total[cl].hist) * 100*u.percent))

            for wr, tot in zip(self.wrong[cl].hist, self.total[cl].hist):
                try:
                    errors = get_efficiency_errors(wr, tot)
                except:
                    errors = [wr/tot if tot > 0 else 0, 0, 0]
                y_eff        [cl].append(errors[0])
                y_eff_lerrors[cl].append(errors[1])
                y_eff_uerrors[cl].append(errors[2])


        plt.style.use('seaborn-talk')
        fig, ax = plt.subplots(4, 2)
        tax = ax[0, 0]
        tax.errorbar(self.wrong["g"].bin_centers(0), y_eff["g"],
                     yerr=[y_eff_lerrors["g"], y_eff_uerrors["g"]])
        tax.set_title("gamma misstag")
        tax.set_xlabel("log(E/GeV)")
        tax.set_ylabel("incorrect / all")

        tax = ax[0,1]
        tax.errorbar(self.wrong["p"].bin_centers(0), y_eff["p"],
                     yerr=[y_eff_lerrors["p"], y_eff_uerrors["p"]])
        tax.set_title("proton misstag")
        tax.set_xlabel("log(E/GeV)")
        tax.set_ylabel("incorrect / all")

        tax = ax[1,0]
        tax.bar(self.total["g"].bin_lower_edges[0][:-1], self.total["g"].hist,
                width=(self.total["g"].bin_lower_edges[0][-1] -
                       self.total["g"].bin_lower_edges[0][0]) /
                len(self.total["g"].bin_centers(0)))
        tax.set_title("gamma numbers")
        tax.set_xlabel("log(E/GeV)")
        tax.set_ylabel("events")

        tax = ax[1,1]
        tax.bar(self.total["p"].bin_lower_edges[0][:-1], self.total["p"].hist,
                width=(self.total["p"].bin_lower_edges[0][-1] -
                       self.total["p"].bin_lower_edges[0][0]) /
                len(self.total["p"].bin_centers(0)))
        tax.set_title("proton numbers")
        tax.set_xlabel("log(E/GeV)")
        tax.set_ylabel("events")

        tax = ax[2, 0]
        tax.hist(right_ratios['g'], bins=20, range=(0, 1), normed=True)
        tax.set_title("fraction of classifiers per event agreeing to gamma")
        tax.set_xlabel("agree ratio")
        tax.set_ylabel("PDF")

        tax = ax[2, 1]
        tax.hist(right_ratios['p'], bins=20, range=(0, 1), normed=True)
        tax.set_title("fraction of classifiers per event agreeing to proton")
        tax.set_xlabel("agree ratio")
        tax.set_ylabel("PDF")

        tax = ax[3, 0]

        histo = np.histogram2d(NTels['g'], proba_ratios['g'],
                               bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
        histo_normed = histo / histo.max(axis=0)
        tax.imshow(histo_normed, interpolation='none', origin='lower', aspect='auto',
                   extent=(1, 9, 0, 1))
        tax.set_title("fraction of classifiers per event agreeing to gamma")
        tax.set_xlabel("NTels")
        tax.set_ylabel("agree ratio")

        tax = ax[3, 1]
        histo = np.histogram2d(NTels['p'], proba_ratios['p'],
                               bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
        histo_normed = histo / histo.max(axis=0)
        tax.imshow(histo_normed, interpolation='none', origin='lower', aspect='auto',
                   extent=(1, 9, 0, 1))
        tax.set_title("fraction of classifiers per event agreeing to proton")
        tax.set_xlabel("NTels")
        tax.set_ylabel("agree ratio")
