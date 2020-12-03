import numpy as np
from bin_spectra import DiscretizeSpectra
import sys
import h5py
import os

import AAKwrapper

import matplotlib.pyplot as plt

import argparse

plt.rcParams.update({'font.size': 16})


class GenerateSingleSpectrum(object):

    def __init__(self, args):
        """
        Initialized with dictionary of default params and 
        """
        
        self.params = self.__default_params()

        self.save_file = args.save_file
        self.save_fig = args.save_fig
        self.fname = args.fname
        self.use_cl = args.use_cl
        self.angle_avg = args.angle_avg
        self.output_phases = args.output_phases
        self.scatter = args.scatter
        self.overwrite = args.overwrite
        

        #reassign initialization param values based on args
        for key in vars(args):
            if key in self.params.keys():

                self.params[key] = vars(args)[key]
                print("Reassigning {}: {}".format(key, self.params[key]))

        
        #initialize spectra binning obj
        if self.angle_avg:
            #use angle averaging
            self.ds = DiscretizeSpectra(self.params, fname=self.fname, \
                    use_cl=self.use_cl, angle_avg=self.angle_avg)
        else:
            #no angle avg
            self.ds = DiscretizeSpectra(self.params, fname=self.fname, \
                    use_cl=self.use_cl)
        
        #output aak phases 
        if self.output_phases: 
            self.t, self.phase_r, self.phase_theta, self.phase_phi, \
            self.omega_r, self.omega_theta, self.omega_phi, self.eccentricity, \
                _ = AAKwrapper.phase(self.params)

        if self.save_fig:
            self.plot_wf()
            self.plot_spectrum()
            

        if self.save_file:
            #make sure data directory exists and if not create it

            if 'data' not in os.listdir("./"):
                os.mkdir('data')

            self.save_data()

    def __default_params(self):
        """
        Defines and returns default parameters when initializing
        """

        #default parameters 
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


    def plot_wf(self):
        """
        Plot generated waveform, save or return
        """

        #plot waveforms
        fig, axs = plt.subplots(2, figsize=(10, 6), sharex=True)
        axs[0].set_title("GW Strain")
        axs[0].plot(self.ds.t, self.ds.h_plus)
        axs[0].set_ylabel("$h_+$")

        axs[1].plot(self.ds.t, self.ds.h_cross)
        axs[1].set_ylabel("$h_x$")
        axs[1].set_xlabel("t")
        fig.tight_layout()

        #save or show
        if self.save_fig:
            plt.savefig("plots/"+self.fname+"_strain.png")
        else:
            return fig

    def plot_spectrum(self):
        """
        Plot generated spectrum
        """
        #make plot for positive frequency data
        mask = self.ds.freqs > 1e-16

        fig = plt.figure(figsize=(10, 6))
        plt.title("Power Spectrum")
        if self.scatter:
            plt.semilogy(self.ds.freqs[mask], self.ds.power[mask]/sum(self.ds.power[mask]), \
                    marker='o', linestyle='None', markersize=3)
        else:
            plt.semilogy(self.ds.freqs[mask], self.ds.power[mask]/sum(self.ds.power[mask]))
        plt.xlim(0, 0.02)
        plt.ylim(1e-5, 1)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("$\\frac{dE}{dt}$ / $\\frac{dE}{dt}_{\mathrm{total}}$")

        #save or show
        if self.save_fig:
            plt.savefig("plots/"+args.fname+"_spec.png")
        else:
            return fig
    
    def save_data(self):
        """
        Save data to file
        """
    
        #check to see if file exists + overwrite if specified
        if self.fname+"_single_wf.hdf5" in os.listdir(os.getcwd()+"/data") and not \
                self.overwrite:
            
            print("file already exists, not overwriting")

        else:

            if self.fname+"_single_wf.hdf5" in os.listdir(os.getcwd()+"/data"):

                print("removing file and overwriting")
                os.remove(os.getcwd()+"/data/"+self.fname+"_single_wf.hdf5")

            #params to output
            output_params = ['e', 'iota', 'p', 'mu', 'M', 's']

            #create output file
            f_wf = h5py.File("data/"+self.fname+"_single_wf.hdf5", "w-")

            print("saving to file:", self.fname+"_single_wf.hdf5")

            #create and populate h5 datasets
            for p in output_params:

                exec("f_wf.create_dataset('{}', data = self.params['{}'])"\
                    .format(p, p, p))

            f_wf.create_dataset('h_plus', data = self.ds.h_plus)
            f_wf.create_dataset('h_cross', data = self.ds.h_cross)
            f_wf.create_dataset('t', data = self.ds.t)
            f_wf.create_dataset('frequencies', data = self.ds.freqs)
            f_wf.create_dataset('power', data = self.ds.power)

        if self.output_phases:

            if self.fname+"_single_phases.hdf5" in os.listdir(os.getcwd()+"/data") \
                    and not self.overwrite:
        
                print("file already exists, not overwriting")

            else:

                if self.fname+"_single_phases.hdf5" in os.listdir(os.getcwd()+"/data"):

                    print("removing file and overwriting")
                    os.remove(os.getcwd()+"/data/"+self.fname+"_single_phases.hdf5")

            
            #phase param names
            output_params = ['t', 'phase_r', 'phase_theta', 'phase_phi', 'omega_r', 'omega_theta', \
                         'omega_phi', 'eccentricity']

            f_phase = h5py.File("data/"+self.fname+"_single_phases.hdf5", "w-")

            print("saving to file:", self.fname+"_single_phases.hdf5")


            for p in output_params:

                exec("f_phase.create_dataset('{}', data = self.{})"\
                .format(p, p, p))



def parser():
    """
    Argument parsing
    """

    #dictionary of AAK wrapper parameter types
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
    parser.add_argument("--use-cl", action='store_true', help="generate wf using command line")
    parser.add_argument("--angle-avg", action='store_true')
    parser.add_argument("--output-phases", action='store_true', help='save phase data along with waveform data')
    parser.add_argument("--scatter", action='store_true', help="generate spectra scatter plot")
    parser.add_argument("--overwrite", action='store_true', help="if data file exists, overwrite")
    parsed, unknown = parser.parse_known_args()

    #accept additional params as AAK EMRI parameters 
    for arg in unknown:

        if arg.startswith(("--", "-")) and (arg.replace('--', '') in param_type_dict.keys() or \
                arg.replace('-', '') in param_type_dict.keys()):

            if arg.startswith("--"):
                key = arg.replace('--', '')
            else:
                key = arg.replace('-', '')

            parser.add_argument(arg, type=param_type_dict[key])

    args = parser.parse_args()
    return args

#############Script################

if __name__ == "__main__":

    args = parser()
    GenerateSingleSpectrum(args)

