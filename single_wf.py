import numpy as np
from bin_spectra import DiscretizeSpectra
import sys
import h5py

import matplotlib.pyplot as plt

import argparse


#12 hours of data with default settings
dt = 60*60*12/2000000

#default parameters 
params = {'backint': False, #end @ plunge and integrate backward
        'LISA': False, #convert to LISA response (Cutler 1998)
        'length': 2000000, #waveform points
        'dt': dt, #time step (s)
        'p': 7.0, #initial semilatus rectum
        'T': 1.166, #waveform dutation (yrs) *if dt unspecified
        'f': 2.e-3, #reference GW freq (Hz) *if p unspecified
        'T_fit': 1., #max duration of local fit (??)
        'mu': 1.e1, #compact object mass (M_sun)
        'M': 1.e6, #BH mass (M_sun)
        's': 0.8, #BH spin (a/M)
        'e': 0.0001, #initial eccentricity
        'iota': 0.0, #inclination angle of L from S
        'gamma': 0., #Initial angle of periapsis from LxS (?)
        'psi': 0., #Initial true anomoly (?)
        'theta_S': 0.785, #source polar angle in ecliptic coords (?)
        'phi_S': 0.785, #source azimuthal angle in ecliptiv coords (?)
        'theta_K': 1.05, #BH spin polar angle
        'phi_K': 1.05, #BH spin azimuthal angle
        'alpha': 0., #initial azimuthal orientation (Eq.(18) in Barack & Cutler, 2004)
        'D': 1.} #source distance (Gpc)



param_type_dict = {'backint': bool,
        'LISA': bool,
        'length': int,
        'dt': float,
        'p': float,
        'T': float,
        'f': float,
        'T_fit': float,
        'mu': float,
        'M': float,
        's': float,
        'e': float,
        'iota': float,
        'gamma': float,
        'psi': float,
        'theta_S': float,
        'phi_S': float,
        'theta_K': float,
        'phi_K': float,
        'alpha': float,
        'D': float}

parser = argparse.ArgumentParser()
parser.add_argument("--save-file", action='store_true', help="save data to file or not")
parser.add_argument("--save-fig", action='store_true', help="save strain and spectra plots")
parser.add_argument("--fname", type=str, help='filename of output')
parser.add_argument("--scatter", action='store_true', help="generate spectra scatter plot")
parsed, unknown = parser.parse_known_args()

#accept additional params as emri params
for arg in unknown:

    if arg.startswith(("--", "-")) and (arg.replace('--', '') in params.keys() or \
            arg.replace('-', '') in params.keys()):

        if arg.startswith("--"):
            key = arg.replace('--', '')
        else:
            key = arg.replace('-', '')

        parser.add_argument(arg, type=param_type_dict[key])

args = parser.parse_args()

#reassign initialization param values based on args
for key in vars(args):
    if key in params.keys():
        
        params[key] = vars(args)[key]
        print("Reassigning {}: {}".format(key, params[key]))

#initialize spectra binning obj
ds = DiscretizeSpectra(params)

print(len(ds.t))

fig, axs = plt.subplots(2, sharex=True)
axs[0].set_title("GW Strain")
axs[0].plot(ds.t, ds.h_plus)
axs[0].set_ylabel("$h_+$")

axs[1].plot(ds.t, ds.h_cross)
axs[1].set_ylabel("$h_x$")
axs[1].set_xlabel("t")
plt.tight_layout()

if args.save_fig:
    plt.savefig("plots/"+args.fname+"_strain.png")
else:
    plt.show()
    plt.close()

#fix this
mask = ds.freqs > 1e-16

plt.figure()
plt.title("Power Spectrum")
#plt.scatter(ds.freqs, ds.power/sum(ds.power))
plt.semilogy(ds.freqs[mask], ds.power[mask]/sum(ds.power[mask]))
plt.xlim(0, 0.02)
plt.ylim(1e-5, 1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")


print(len(ds.freqs[mask]))

if args.save_fig:
    plt.savefig("plots/"+args.fname+"_spec.png")
else:
    plt.show()
    plt.close()

if args.scatter:

    mask = ds.freqs > 1e-16

    plt.figure()
    plt.title("Power Spectrum")
    #plt.scatter(ds.freqs, ds.power/sum(ds.power))
    plt.semilogy(ds.freqs[mask], ds.power[mask]/sum(ds.power[mask]), marker='o', linestyle='None', markersize=0.8)
    plt.xlim(0, 0.02)
    plt.ylim(1e-5, 1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")

    if args.save_fig:
        plt.savefig("plots/"+args.fname+"_spec_scatter.png") 
    else:
        plt.show()
        plt.close()

if args.save_file:

    output_params = ['e', 'iota', 'p', 'mu', 'M', 's']

    f = h5py.File("data/"+args.fname+"_single_wf.hdf5", "w-")


    for p in output_params:

        exec("f.create_dataset('{}', data = params['{}'])"\
                .format(p, p, p))

    f.create_dataset('h_plus', data = ds.h_plus)
    f.create_dataset('h_cross', data = ds.h_cross)
    f.create_dataset('t', data = ds.t)
    f.create_dataset('frequencies', data = ds.freqs)
    f.create_dataset('power', data = ds.power)
