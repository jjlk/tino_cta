#!/usr/bin/env python3

import numpy as np

from astropy.table import Table
from astropy import units as u

from itertools import chain

from helper_functions import *

from ctapipe.analysis.Sensitivity import *
from ctapipe.analysis.Sensitivity import crab_source_rate, CR_background_rate, Eminus2

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
#plt.style.use('t_slides')

'''
MC energy ranges:
gammas: 0.1 to 330 TeV
proton: 0.1 to 600 TeV
'''
edges_gammas = np.linspace(2, np.log10(330000), 28)
edges_proton = np.linspace(2, np.log10(600000), 30)

angle_unit = u.deg
energy_unit = u.GeV
flux_unit = (u.erg*u.cm**2*u.s)**(-1)
sensitivity_unit = flux_unit * u.erg**2

observation_time = 3.6*u.h

# PyTables
import tables as tb
# pandas data frames
import pandas as pd

def open_pytable_as_pandas(filename, mode='r'):
    pyt_infile = tb.open_file(filename, mode=mode)
    pyt_table = pyt_infile.root.reco_events

    return pd.DataFrame(pyt_table[:])

apply_cuts = True
def selection_mask(event_table, ntels=2, gammaness=.50):
    return (event_table["NTels_reco"] >= ntels) & \
           (event_table["gammaness"] > gammaness)
           #& (event_table["off_angle"] < .3)
         #& (event_table["gamma_ratio"] > gamma_ratio), gamma_ratio=.75


if __name__ == "__main__":

    parser = make_argparser()
    parser.add_argument('--events_dir', type=str, default="data/reconstructed_events")
    parser.add_argument('--in_file', type=str, default="classified_events")
    args = parser.parse_args()

    NReuse_Gammas = 10
    NReuse_Proton = 20

    NGammas_per_File = 5000 * NReuse_Gammas
    NProton_per_File = 5000 * NReuse_Proton

    NGammas_simulated = NGammas_per_File * 85
    NProton_simulated = NProton_per_File * (3000-100)

    print()
    print("gammas simulated:", NGammas_simulated)
    print("proton simulated:", NProton_simulated)
    print()
    print("observation time:", observation_time)

    #gammas = open_pytable_as_pandas(
            #"{}/{}_{}_{}.h5".format(args.events_dir, args.in_file, "gamma", "wave"))

    #proton = open_pytable_as_pandas(
            #"{}/{}_{}_{}.h5".format(args.events_dir, args.in_file, "proton", "wave"))

    #applying some cuts
    #if apply_cuts:
        #gammas = gammas[selection_mask(gammas)]
        #proton = proton[selection_mask(proton)]
    #print("gammas selected (wavelets):", len(gammas))
    #print("proton selected (wavelets):", len(proton))

    #SensCalc = Sensitivity_PointSource(
                        #mc_energies={'g': gammas['MC_Energy'],
                                     #'p': proton['MC_Energy']},
                        #off_angles={'g': gammas['off_angle'].values*angle_unit,
                                    #'p': proton['off_angle'].values*angle_unit},
                        #energy_bin_edges={'g': edges_gammas,
                                        #'p': edges_proton},
                        #energy_unit=energy_unit, flux_unit=flux_unit)

    #Eff_Areas = SensCalc.get_effective_areas(
                    #n_simulated_events={'g': NGammas_simulated,
                                        #'p': NProton_simulated},
                    #generator_spectra={'g': Eminus2, 'p': Eminus2},
                    #generator_areas={'g': np.pi * (1000*u.m)**2,
                                     #'p': np.pi * (2000*u.m)**2})

    #exp_events_per_E = SensCalc.get_expected_events(
        #rates={'g': crab_source_rate, 'p': CR_background_rate},
        #observation_time=observation_time)

    #NExpGammas = sum(exp_events_per_E['g'])
    #NExpProton = sum(exp_events_per_E['p'])

    #print()
    #print("expected gammas (wavelets):", NExpGammas)
    #print("expected proton (wavelets):", NExpProton)

    #if args.plot and args.verbose:
        #plt.figure()
        #plt.semilogy((edges_gammas[1:] + edges_gammas[:-1])/2,
                     #Eff_Areas['g'], "b", label='Gammas')
        #plt.semilogy((edges_proton[1:] + edges_proton[:-1])/2,
                     #Eff_Areas['p'], "r", label='Protons')
        #plt.title("Effective Area")
        #plt.xlabel(r"$\log_{10}(E/\mathrm{GeV})$")
        #plt.ylabel(r"$A_\mathrm{eff} / \mathrm{m}^2$")
        #plt.pause(.1)

    #weights = SensCalc.scale_events_to_expected_events()
    #sensitivities = SensCalc.get_sensitivity(
                                #sensitivity_energy_bin_edges=np.linspace(2, 6, 17))

    # now for tailcut
    gammas_t = open_pytable_as_pandas(
            "{}/{}_{}_{}.h5".format(args.events_dir, args.in_file, "gamma", "tail"))

    proton_t = open_pytable_as_pandas(
            "{}/{}_{}_{}.h5".format(args.events_dir, args.in_file, "proton", "tail"))

    # applying some cuts
    if apply_cuts:
        print("gammas, protons before cuts (tailcuts):", len(gammas_t), len(proton_t))
        gammas_t = gammas_t[selection_mask(gammas_t)]
        proton_t = proton_t[selection_mask(proton_t)]
        print("gammas, protons after cuts (tailcuts):", len(gammas_t), len(proton_t))



    SensCalc_t = Sensitivity_PointSource(
                    mc_energies={'g': gammas_t['MC_Energy'],
                                 'p': proton_t['MC_Energy']},
                    off_angles={'g': gammas_t['off_angle']*angle_unit,
                                'p': proton_t['off_angle']*angle_unit},
                    energy_bin_edges={'g': edges_gammas,
                                      'p': edges_proton}, verbose=True,
                    energy_unit=energy_unit, flux_unit=flux_unit)

    sensitivities_t = SensCalc_t.calculate_sensitivities(
                            #generator_energy_hists=SensCalc.generator_energy_hists,
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_spectra={'g': Eminus2, 'p': Eminus2},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            rates={'g': crab_source_rate,
                                   'p': CR_background_rate},
                            sensitivity_energy_bin_edges=np.linspace(2, 6, 17))
    weights_t = SensCalc_t.event_weights

    print()
    print("gammas selected (tailcuts):", len(gammas_t))
    print("proton selected (tailcuts):", len(proton_t))

    NExpGammas_t = sum(SensCalc_t.exp_events_per_energy_bin['g'])
    NExpProton_t = sum(SensCalc_t.exp_events_per_energy_bin['p'])

    print()
    print("expected gammas (tailcuts):", NExpGammas_t)
    print("expected proton (tailcuts):", NExpProton_t)

    # do some plotting
    if args.plot:
        bin_centres_g = (edges_gammas[1:]+edges_gammas[:-1])/2.
        bin_centres_p = (edges_proton[1:]+edges_proton[:-1])/2.

        plt.figure()
        plt.loglog(
                10**bin_centres_g,
                SensCalc_t.exp_events_per_energy_bin['g'], label="gamma")
        plt.loglog(
                10**bin_centres_p,
                SensCalc_t.exp_events_per_energy_bin['p'], label="proton")
        plt.ylabel("expected events in {}".format(observation_time))
        plt.legend()

        # plot effective area
        plt.figure(figsize=(16,8))
        plt.suptitle("ASTRI Effective Areas")
        plt.subplot(121)
        #plt.loglog(
            #10 ** bin_centres_g,
            #SensCalc.effective_areas['g'], label="wavelets")
        plt.loglog(
            10 ** bin_centres_g,
            SensCalc_t.effective_areas['g'], label="tailcuts")
        plt.xlabel(r"$E_\mathrm{MC} / \mathrm{GeV}$")
        plt.ylabel(r"effective area / $\mathrm{m^2}$")
        plt.title("gammas")
        plt.legend()

        plt.subplot(122)
        #plt.loglog(
            #10 ** bin_centres_p,
            #SensCalc.effective_areas['p'], label="wavelets")
        plt.loglog(
            10 ** bin_centres_p,
            SensCalc_t.effective_areas['p'], label="tailcuts")
        plt.xlabel(r"$E_\mathrm{MC} / \mathrm{GeV}$")
        plt.ylabel(r"effective area / $\mathrm{m^2}$")
        plt.title("protons")
        plt.legend()

        plt.pause(.1)

        # the point-source sensitivity binned in energy
        plt.figure()
        #plt.semilogy(
            #sensitivities["MC Energy"],
            #(sensitivities["Sensitivity"].to(flux_unit) *
             #sensitivities["MC Energy"].to(u.erg)**2),
            #color="darkred",
            #marker="s",
            #label="wavelets")
        #plt.semilogy(
            #sensitivities["MC Energy"].to(u.TeV),
            #(sensitivities["Sensitivity_uncorr"].to(flux_unit) *
             #sensitivities["MC Energy"].to(u.erg)**2),
            #color="darkgreen",
            #marker="^",
            #ls="",
            #label="wavelets (no upscale)")

        plt.semilogy(
            sensitivities_t["MC Energy"].to(u.TeV),
            (sensitivities_t["Sensitivity"].to(flux_unit) *
             sensitivities_t["MC Energy"].to(u.erg)**2),
            color="C0",
            marker="s",
            label="tailcuts")
        plt.semilogy(
            sensitivities_t["MC Energy"].to(u.TeV),
            (sensitivities_t["Sensitivity_uncorr"].to(flux_unit) *
             sensitivities_t["MC Energy"].to(u.erg)**2),
            color="darkorange",
            marker="^",
            ls="",
            label="tailcuts (no upscale)")
        plt.legend()
        plt.xlabel('E / {:latex}'.format(u.TeV))
        plt.ylabel(r'$E^2 \Phi /$ {:latex}'.format(sensitivity_unit))
        plt.gca().set_xscale("log")
        plt.xlim([1e-2, 2e3])
        plt.pause(.1)

        # plot a sky image of the events
        # useless since too few actual background events
        if False:
            fig2 = plt.figure()
            plt.hexbin(
                [(ph-180)*np.sin(th*u.deg) for
                    ph, th in zip(chain(gammas['phi'], proton['phi']),
                                  chain(gammas['theta'], proton['theta']))],
                [a for a in chain(gammas['theta'], proton['theta'])],
                gridsize=41, extent=[-2, 2, 18, 22],
                C=[a for a in chain(weights['g'], weights['p'])],
                bins='log'
                )
            plt.colorbar().set_label("log(Number of Events)")
            plt.axes().set_aspect('equal')
            plt.xlabel(r"$\sin(\vartheta) \cdot (\varphi-180) / ${:latex}"
                       .format(angle_unit))
            plt.ylabel(r"$\vartheta$ / {:latex}".format(angle_unit))
            if args.write:
                save_fig("plots/skymap")

        # plot the angular distance of the reconstructed shower direction
        # from the pseudo-source
        figure = plt.figure()
        bins = 60

        plt.subplot(211)
        plt.hist([
                  #1-np.cos(proton_t['off_angle']*u.deg.to(u.rad)),
                  proton_t['off_angle']**2,
                  gammas_t["off_angle"]**2],
                 weights=[weights_t['p'], weights_t['g']],
                 rwidth=1, stacked=True,
                 range=(0, .3), label=("protons", "gammas"),
                 log=False, bins=bins)
        plt.xlabel(r"$(\vartheta/^\circ)^2$")
        #plt.xlabel(r"$1-\cos(\vartheta)$")
        plt.ylabel("expected events in {}".format(observation_time))
        plt.xlim([0, .3])
        plt.title("tailcuts")
        plt.legend()


        #plt.subplot(212)
        #if True:
            #NProtons = np.sum(proton['off_angle'][(proton['off_angle'].values**2) < 10])
            #proton_weight_flat = np.ones(bins) * NProtons/bins
            #proton_angle_flat = np.linspace(0, 10, bins, False)
            #proton_angle = proton_angle_flat
            #proton_weight = proton_weight_flat
        #else:
            #proton_angle = proton['off_angle']**2
            #proton_weight = weights['p']

        #plt.hist([proton_angle,
                  #gammas['off_angle'].values**2],
                 #weights=[proton_weight, weights['g']], rwidth=1, stacked=True,
                 #range=(0, 10), label=("protons", "gammas"),
                 #log=True, bins=bins)
        #plt.xlabel(r"$\vartheta^2 / \mathrm{"+str(angle_unit)+"}^2$")
        #plt.ylabel("expected events in {}".format(SensCalc.observation_time))
        #plt.ylim([1e-1, 1e6])
        #plt.title("wavelets")
        #plt.legend()

        if args.write:
            save_fig("plots/theta_square")

        plt.pause(.1)
        #plt.subplot(313)
        #plt.hist([1-np.clip(np.cos(proton['off_angles']*u.degree.to(u.rad)), -1, 1-1e-6),
                  #1-np.clip(np.cos(gammas['off_angles']*u.degree.to(u.rad)), -1, 1-1e-6)],
                 #weights=[weight_p, weight_g], rwidth=1, stacked=True,
                 #range=(0, 5e-3),
                 #bins=30)
        #plt.xlabel(r"$1-\cos(\vartheta)$")
        #plt.ylabel("expected events in {}".format(SensCalc.observation_time))
        #plt.xlim([0, 5e-3])

        #plt.tight_layout()

        #figure = plt.figure()
        #if True:
            #NProtons_t = np.sum(proton_t['off_angles'][(proton_t['off_angles']**2) < 10])
            #proton_weight_flat = np.ones(50) * NProtons_t/50
            #proton_angle_flat = np.linspace(0,10,50,False)
            #proton_angle_t = proton_angle_flat
            #proton_weight_t = proton_weight_flat
        #else:
            #proton_angle_t = proton_t['off_angles']**2
            #proton_weight_t = weights_t['p']

        #plt.hist([proton_angle_t,
                  #gammas_t['off_angles']**2],
                 #weights=[proton_weight_t, weights_t['g']], rwidth=1, stacked=True,
                 #range=(0, 10), label=("protons", "gammas"),
                 #log=True, bins=50)
        #plt.xlabel(r"$\vartheta^2 / \mathrm{"+str(angle_unit)+"}^2$")
        #plt.ylabel("expected events in {}".format(SensCalc.observation_time))
        #plt.ylim([1e-1, 1e5])
        #plt.legend()
        #plt.suptitle("tail cuts (10 PE / 5 PE)")


        plt.show()





