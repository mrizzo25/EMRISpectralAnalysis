import numpy as np
import matplotlib.pyplot as plt
import h5py

from bin_spectra import DiscretizeSpectra

import argparse

########### Definitions ############

#12 hours of data with default settings
dt = 60*60*12/2000000

#default params
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

param_grid_ranges = {'e': [0.0001, 0.9], 
                     'iota': [0, 80*np.pi/180.],
                     'mu': [10, 100], 
                     'M': [1e6, 1e8], #might want to make this log valued
                     's': [0.001, 0.90]}


########### Main Class ##############

class GenerateSpectra(object):

    def __init__(self, args):

        self.params = params

        self.n_grid = args.n_grid
        self.grid_param = args.grid_param
        
        self.binning = args.binning
        self.power_cutoff = args.power_cutoff
        
        self.fname = args.fname
        self.existing = args.existing

        self.verbose = args.verbose

        if self.binning:
            self.n_bins = args.n_bins


        #file to store data
        #actually implement append feature at some point
        if self.existing:
            self.f = h5py.File("data/"+self.fname+".hdf5", "r+")
        else:
            self.f = h5py.File("data/"+self.fname+".hdf5", "w-")


        #reassign initialization param values based on args
        for key in vars(args):
            if key in self.params.keys():
                self.params[key] = vars(args)[key]

                if self.verbose:
                    print("Reassigning {}: {}".format(key, self.params[key]))


        #initialize bins
        if self.binning:
            #initialize spectra binning obj
            self.ds = DiscretizeSpectra(self.params, self.n_bins, \
                    power_cutoff=self.power_cutoff)
            _, self.bins = self.ds.bin_spec()

        else:
            self.ds = DiscretizeSpectra(self.params, power_cutoff=self.power_cutoff)

        #regardless of binning, store frequencies
        self.freqs = self.ds.freqs

        #run initialization and grid population methods
        self.initialize()
        self.evaluate_grid()


    def initialize(self):
        """
        Set up parameter grid and output file datasets
        """
        #total number of grid points
        n_out = self.n_grid**len(self.grid_param)

        if self.verbose:
            print("Total number of grid pts: {}".format(n_out))

        #make list of parameter pairs to iterate over
        for g in self.grid_param:

            #1D list
            exec("{}_vals = np.linspace({:f}, {:f}, {:d})"\
                .format(g, param_grid_ranges[g][0], \
                param_grid_ranges[g][1], self.n_grid))
            
            #h5py dataset output
            exec("self.{}_dset = self.f.create_dataset('{}', ({:d},), dtype={})"\
                .format(g, g, n_out, \
                param_type_dict[g].__name__))
    
            if self.verbose:
                print("Creating grid data for: {}".format(g))

            
        exec("{} = np.meshgrid({})"\
            .format(', '.join([p.upper() for p in self.grid_param]), \
                    ', '.join([p+'_vals' for p in self.grid_param])))

        #lol
        exec("self.param_set = np.array(list(map(list, zip({}))))"\
            .format(', '.join([p.upper()+'.ravel()' for p in self.grid_param])))

        #create other h5 datasets 
        if self.binning:
            self.bin_dset = self.f.create_dataset("bin_edges", data = self.bins)
            self.hist_dset = self.f.create_dataset("hist", (n_out, self.n_bins))
            
        self.freq_dset = self.f.create_dataset("frequencies", data=self.freqs)
        self.power_dset = self.f.create_dataset("power", (n_out, len(self.freqs)), \
                dtype=complex)


    def evaluate_grid(self):
        """
        Generate spectra - binned or otherwise - for each point on the grid
        """
        grid_pt_dict = {}

        for i in range(len(self.param_set)):
            
            #dict of param values at a given grid point
            grid_pt_dict = dict(zip(self.grid_param, self.param_set[i, :]))

            if self.verbose:
                print("Param vals:", grid_pt_dict)

            #there's probably a cleaner way to do these next few lines...
            if self.binning:
                try:
                    self.ds.change_parms(grid_pt_dict)
                    hist, _ = self.ds.bin_spec()
                    power = self.ds.power

                except:

                    if self.verbose:
                        print("Point failed")

                    hist = np.zeros(self.n_bins)
                    power = np.zeros(len(self.freqs))

                self.hist_dset[i] = hist

            else:
                try:
                    self.ds.change_params(grid_pt_dict)
                    power = self.ds.power

                except:
                    if self.verbose:
                        print("Point failed")


                    power = np.zeros(len(self.freqs))

            self.power_dset[i] = power

            #add parameter values to dataset
            for key in grid_pt_dict:
                exec("self.{}_dset[{:d}] = {}".format(key, i, grid_pt_dict[key]))


########## Parser ##################

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help='filename of output')
parser.add_argument("--existing", action='store_true', help="append to existing file")
parser.add_argument("--grid-param", type=str, action="append", help="param to grid over")
parser.add_argument("--binning", action='store_true', help="whether or not to bin spectra")
#TODO: make this variable
parser.add_argument("--n-grid", type=int, help="number of points for each param on grid")
parser.add_argument("--n-bins", type=int, default=500, help="number of spectral bins (evenly spaced), defaults to 500")
parser.add_argument("--power-cutoff", type=float, default=-8, help="log cutoff point for power spectrum (powers smaller than this will be discarded)")
parser.add_argument("--verbose", action='store_true', help="print extra debugging output")
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

args=parser.parse_args()

########## Script ################## 

GenerateSpectra(args)
