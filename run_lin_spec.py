import numpy as np
import matplotlib.pyplot as plt
import h5py

from bin_spectra import DiscretizeSpectra

#to be implemented: command-line options
#import argparse

#file to store data
f = h5py.File("wf_spec_data.hdf5", "w")

#12 hours of data with default settings
dt = 60*60*12/2000000

#default parameters 
pars = {'backint': False, #end @ plunge and integrate backward
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

#number of values for each range
n = 10
n_bins = 100

#initialize spectra binning obj
ds = DiscretizeSpectra(pars, n_bins, power_cutoff=-8)

#compute reference spectrum
hist, bins = ds.bin_spec()

#define ranges of eccentricity and inclination values
e_vals = np.linspace(0.0001, 0.9, n)
i_vals = np.linspace(0, 80, n) * np.pi/180.

#initialize datasets
e_dset = f.create_dataset("eccentricity", (n**2+1,), dtype=float)
i_dset = f.create_dataset("inclination", (n**2+1,), dtype=float)
hist_dset = f.create_dataset("hist", (n**2+1, n_bins), dtype=float)

#populate bins (should be the same for all)
bin_dset = f.create_dataset("bin_edges", data=bins)

#populate frequencies
freq_dset = f.create_dataset("frequencies", data=ds.freqs)

power_dset = f.create_dataset("power", (n**2+1, len(ds.freqs)), dtype=complex)

#first values
e_dset[0] = 0.0001
i_dset[0] = 0.0
hist_dset[0] = hist
power_dset[0] = ds.power

#storage index
index=1

for e in e_vals:
    for i in i_vals:

        print('e:', e, "iota:", i)

        #for each combo of values, compute histogram
        try:
            ds.change_params({'e':e, 'iota':i})
            hist, _ = ds.bin_spec()
        except:
            hist = np.zeros(n)

        e_dset[index] = e
        i_dset[index] = i
        hist_dset[index] = hist
        power_dset[index] = ds.power
    
        index+=1
