import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

#astrophysical constants (for conversion)
import astropy.constants as const
G_SI = const.G.value
C_SI = const.c.value

import traceback

from GenerateWF import WFGen

import argparse

########### Main Class ##############

class GenerateSpectra(object):

    def __init__(self, args):
        """
        Takes argparse object *or* class with 
        the appropriate attributes if running from an external script

        see commandline --help for parameter definitions
        """

        #local instances of params + grid ranges
        self.params = self.__default_params()
        self.param_grid_ranges = self.__default_grid_ranges()

        self.grid_param = args.grid_param
        
        #same number of pts for each param
        if len(args.n_grid) == 1:
            self.n_grid = list(np.ones(len(self.grid_param), dtype=int) * args.n_grid[0])
        
        #different number for each param
        else:
            
            #check to make sure len matches number of parameters
            if len(args.n_grid) != len(self.grid_param):
                raise AssertionError("Number of grid dimensions needs to match \
                        number of grid parameters")
            
            self.n_grid = args.n_grid
       

        #which params to log scale, if any
        self.log_scale_param = args.log_scale_param
        
        if self.p_isco_fraction is not None:
            self.p_isco_fraction = float(self.p_isco_fraction)

        if self.p_mb_fraction is not None:
            self.p_mb_fraction = float(self.p_mb_fraction)

        if self.fix_sma is not None:
            self.fix_sma = float(self.fix_sma)

        self.p_isco_fraction = args.p_isco_fraction
        self.p_mb_fraction = args.p_mb_fraction
        self.fix_sma = args.fix_sma

        self.store_wf = args.store_wf
        self.power_cutoff = args.power_cutoff
        
        self.fname = args.fname
        self.existing = args.existing
        self.use_cl = args.use_cl

        self.verbose = args.verbose
        self.overwrite = args.overwrite


        #file to store data
        #actually implement append feature at some point
        if self.existing:
            self.f = h5py.File("data/"+self.fname+".hdf5", "r+")
        else:
            if self.fname+".hdf5" in os.listdir("data"):
                if self.overwrite:
                    os.remove("data/"+self.fname+".hdf5")
                    self.f = h5py.File("data/"+self.fname+".hdf5", "w-")
                else:
                    raise FileExistsError("Please pick different file name, \
                            or enable overwrite")
            else:
                self.f = h5py.File("data/"+self.fname+".hdf5", "w-")


        #reassign initialization param values based on args
        for key in vars(args):
            if key in self.params.keys():
                self.params[key] = vars(args)[key]

                if self.verbose:
                    print("Reassigning {}: {}".format(key, self.params[key]))
        

        #take care of param_grid range adjustments
        for key in vars(args):

            #reassign min in param range
            if key.endswith("max") and key.split('_')[0] in self.param_grid_ranges.keys():

                p = key.split('_')[0]
                self.param_grid_ranges[p][1] = vars(args)[key]

                if self.verbose:

                    print("Reassigning", p, "max:", vars(args)[key])

            #reassign max in param range
            elif key.endswith("min") and key.split('_')[0] in self.param_grid_ranges.keys():

                p = key.split('_')[0]
                self.param_grid_ranges[p][0] = vars(args)[key]

                if self.verbose:

                    print("Reassigning", p, "min:", vars(args)[key])

        
        self.ds = DiscretizeSpectra(self.params, self.fname, \
                    use_cl=self.use_cl, power_cutoff=self.power_cutoff)
            
        if self.verbose:
            print("Spectra generation module initialized")

        #regardless of binning, store frequencies
        self.freqs = self.ds.freqs

        #if storing waveforms, store time values
        if self.store_wf:
            self.times = self.ds.t

        #run initialization and grid population methods
        self.initialize()
        self.evaluate_grid()

    def __default_params(self):
        """
        Creates dict of default AAK params
        """
            
        #default params
        params = {'backint': False, #end @ plunge and integrate backward
                'LISA': False, #convert to LISA response (Cutler 1998)
                'length': 2000000, #waveform points
                'dt': 0.0216, #time step (s)
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

        return params

    def __default_grid_ranges(self):
        """
        Create dictionary of parameter grid ranges
        """

        param_grid_ranges = {'e': [0.0001, 0.9], 
                     'iota': [0, 80*np.pi/180.],
                     'mu': [10, 100], 
                     'M': [1e5, 1e7], 
                     's': [0.001, 0.90], 
                     'piscofrac': [1.5, 3],
                     'pmbfrac': [1.5, 3],
                     'p': [6.0, 10.0]}

        return param_grid_ranges

    def initialize(self):
        """
        Set up parameter grid and output file datasets
        """
        #total number of grid points
        n_out = np.prod(self.n_grid)

        if self.verbose:
            print("Total number of grid pts: {}".format(n_out))

        #make list of parameter pairs to iterate over
        for g, n in zip(self.grid_param, self.n_grid):

            #1D list
            if self.log_scale_param is not None:
                if g in self.log_scale_param:
                    if self.verbose:
                        print("Generating log grid in", g)
                
                    exec("{}_vals = np.logspace({:f}, {:f}, {:d})"\
                        .format(g, np.log10(self.param_grid_ranges[g][0]), \
                        np.log10(self.param_grid_ranges[g][1]), n))
            
                else:
                    if self.verbose:
                        print("Generating grid in", g)

                    exec("{}_vals = np.linspace({:f}, {:f}, {:d})"\
                        .format(g, self.param_grid_ranges[g][0], \
                        self.param_grid_ranges[g][1], n))
            else:
                if self.verbose:
                    print("Generating grid in", g)

                exec("{}_vals = np.linspace({:f}, {:f}, {:d})"\
                    .format(g, self.param_grid_ranges[g][0], \
                    self.param_grid_ranges[g][1], n))


            #h5py dataset output
            exec("self.{}_dset = self.f.create_dataset('{}', ({:d},), dtype=float)"\
                .format(g, g, n_out))
    
            if self.verbose:
                print("Creating grid data for: {}".format(g))

            
        exec("{} = np.meshgrid({})"\
            .format(', '.join([p.upper() for p in self.grid_param]), \
                    ', '.join([p+'_vals' for p in self.grid_param])))

        #create an array of parameter set values at every grid point
        exec("self.param_set = np.array(list(map(list, zip({}))))"\
            .format(', '.join([p.upper()+'.ravel()' for p in self.grid_param])))

            
        self.freq_dset = self.f.create_dataset("frequencies", data=self.freqs)
        self.power_dset = self.f.create_dataset("power", (n_out, len(self.freqs)), \
                dtype=complex)
        
        if self.store_wf:
            self.times = self.f.create_dataset("times", data=self.times)
            self.hplus_dset = self.f.create_dataset("h_plus", \
                    (n_out, len(self.times)), dtype=float)
            self.hcross_dset = self.f.create_dataset("h_cross", \
                    (n_out, len(self.times)), dtype=float)


    def evaluate_grid(self):
        """
        Generate spectra - binned or otherwise - for each point on the grid
        """
        grid_pt_dict = {}

        for i in range(len(self.param_set)):
            
            #dict of param values at a given grid point
            grid_pt_dict = dict(zip(self.grid_param, self.param_set[i, :]))

            reassign_params = grid_pt_dict.copy()

            if self.verbose:
                print("Param vals:", grid_pt_dict)
                  

            #option to fix p/p_isco ratio (only matters if varying spin)
            #also valid of piscofrac is a grid parameter
            if self.p_isco_fraction is not None or 'piscofrac' in self.grid_param:

                reassign_params = self.fractional_p_isco(reassign_params, grid_pt_dict)

            elif self.p_mb_fraction is not None or 'pmbfrac' in self.grid_param:

                reassign_params = self.fractional_p_mb(reassign_params, grid_pt_dict)

            elif self.fix_sma is not None of 'fixsma' in self.grid_param:

                reassign_params = self.fix_semimajor_axis(reassign_params, grid_pt_dict)
           

        
            #try changing parameters of waveform and regenerating
            try:
                    
                self.ds.change_params(reassign_params, fname=self.fname+str(i), \
                            use_cl=self.use_cl)
                power = self.ds.power

                if self.store_wf:
                    h_plus = self.ds.h_plus
                    h_cross = self.ds.h_cross

            except:

                if self.verbose:
                    traceback.print_exc()
                    print("Point failed")

                power = np.zeros(len(self.freqs), dtype=complex)
            
                if self.store_wf:
                    h_cross = np.zeros(len(self.times))
                    h_plus = np.zeros(len(self.times))


            self.power_dset[i] = power
        
            if self.store_wf:

                self.hplus_dset[i] = h_plus
                self.hcross_dset[i] = h_cross

            #add parameter values to dataset
            for key in grid_pt_dict:
                
                exec("self.{}_dset[{:d}] = {}".format(key, i, grid_pt_dict[key]))

    def fractional_p_isco(self, reassign_params, grid_pt_dict):
        
        #if 's' not a grid parameter, use the static value
        if 's' in self.grid_param:
            s = grid_pt_dict['s']
        else:
            s = self.params['s']

        #equations grabbed from wikipedia
        z1 = 1 + (1 - s**2)**(1/3) * \
                ((1 + s)**(1/3) + \
                (1 - s)**(1/3))

        z2 = np.sqrt(3 * s**2 + z1**2)

        #gonna assume everything is prograde
        p_isco = (3 + z2 - np.sqrt((3. - z1) * (3. + z1 + 2. * z2)))

        #assign according to grid parameter
        if 'piscofrac' in self.grid_param:

            reassign_params['p'] = grid_pt_dict['piscofrac'] * p_isco

            reassign_params.pop('piscofrac')

            if self.verbose:
                print("Setting p =", reassign_params['p'], "for p_isco_frac", grid_pt_dict['piscofrac'])

        #or else use the fixed value
        else:
            
            reassign_params['p'] = self.p_isco_fraction * p_isco

            if self.verbose:
                print("Setting p =", reassign_params['p'], "for p_isco_frac", self.p_isco_fraction)

        return reassign_params

    def fractional_p_mb(self, reassign_params, grid_pt_dict):

        #if 's' not a grid parameter, use the static value
        if 's' in self.grid_param:
            s = grid_pt_dict['s']
        else:
            s = self.params['s']

        p_mb = 2 - s + 2 * (1 - s)**0.5

        if 'pmbfrac' in self.grid_param:
            reassign_params['p'] = grid_pt_dict['pmbfrac'] * p_mb

            reassign_params.pop('pmbfrac')

            if self.verbose:
                print("Setting p =", reassign_params['p'], "for p_mb_frac", grid_pt_dict['pmbfrac'])

        #or else use the fixed value
        else:
            reassign_params['p'] = self.p_mb_fraction * p_isco

            if self.verbose:
                print("Setting p =", reassign_params['p'], "for p_mb_frac", self.p_mb_fraction)

        return reassign_params    


    def fix_semimajor_axis(self, reassign_params, grid_pt_dict):

        if 'e' in self.grid_param:
            e = grid_pt_dict['e']
        else:
            e = self.params['e']

        if 'fixsma' in self.grid_param:
        
            reassign_params['p'] = grid_pt_dict['fixsma'] * (1. - e**2)

            if self.verbose:
                print("Setting p =", reassign_params['p'], "for fixed sma", grid_pt_dict['fix_sma'])


        else:

            reassign_params['p'] = self.fix_sma * (1. - e**2)

            if self.verbose:
                print("Setting p =", reassign_params['p'], "for fixed sma", self.fix_sma)

        return reassign_params


def parser():
    """
    Argument parsing
    """

    #dictionary of AAK wrapper param types
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
        'D': float, 
        'piscofrac': float,
        'pmbfrac': float, 
        'fixsma': float}


    ########## Parser ##################

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, help='filename of output')
    parser.add_argument("--existing", action='store_true', help="append to existing file")
    parser.add_argument("--use-cl", action='store_true', help="use command line waveform generation")
    parser.add_argument("--grid-param", type=str, action="append", help="param to grid over")
    parser.add_argument("--log-scale-param", type=str, action="append", help="use log spacing for this params on grid")
    
    parser.add_argument("--p-isco-fraction", default=None, help="fix slr to fractional value of isco slr")
    parser.add_argument("--p-mb-fraction", default=None, help="fix value of marginally bound radius")
    parser.add_argument("--fix-sma", default=None, help="fix semimajor axis")
    
    parser.add_argument("--store-wf", action='store_true', help="save gw polarizations")
    parser.add_argument("--n-grid", nargs="+", type=int, help="number of points for each param on grid")
    parser.add_argument("--power-cutoff", type=float, default=-8, help="log cutoff point for power spectrum (powers smaller than this will be discarded)")
    parser.add_argument("--filter", default=None, type=str, help="Not implemented yet: functions to filter points ('band_cut', 'snr_cut')")
    parser.add_argument("--overwrite", action='store_true', help="if output data file already exists, overwrite it")
    parser.add_argument("--verbose", action='store_true', help="print extra debugging output")
    parsed, unknown = parser.parse_known_args() 

    #accept additional params as emri params
    for arg in unknown:
    
        if arg.startswith(("--", "-")) and (arg.replace('--', '') in param_type_dict.keys() or \
                arg.replace('-', '') in param_type_dict.keys()):

            if arg.startswith("--"):
                key = arg.replace('--', '')
            else:
                key = arg.replace('-', '')

            parser.add_argument(arg, type=param_type_dict[key])

        #accept params of the form "param-max" and "param-min"
        elif arg.endswith("max") and arg.replace('--', '').split('-')[0] in param_type_dict.keys():

            parser.add_argument(arg, type=float)

        elif arg.endswith("min") and arg.replace('--', '').split('-')[0] in param_type_dict.keys():
        
            parser.add_argument(arg, type=float)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    ########## Script ################## 

    args = parser()
    GenerateSpectra(args)
