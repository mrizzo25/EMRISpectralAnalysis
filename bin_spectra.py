#Monica Rizzo, 2020
import numpy as np
from generate_spectra import compute_fft


class DiscretizeSpectra(object):
    """
    compute and bin spectrum of an EMRI waveform snapshot
    """

    def __init__(self, params, bins, bin_range=[0.0, 0.014], power_cutoff=-5.):
        """
        params: dict of params handed to AAK solver
        bins: number of frequency bins
        bin_range: range of histogram
        power_cutoff: exponent of cutoff power
        """

        #create private parameter attribute
        self._params = params
        
        self.bins = bins
        self.bin_range = bin_range
        self.power_cutoff = power_cutoff


        #compute waveform and fft and store locally 
        self.h_plus, self.h_cross, self.t, \
                self.power, self.freqs = compute_fft(self._params)

    def change_params(self, change_params):
        """
        change_params: dict of parameters to change 
                    before recomputing wf
        """

        #this should be a dictionary
        #make sure the keys match up

        if isinstance(change_params, dict):
            for key in change_params.keys():
                self._params[key] = change_params[key]
        else:
            raise TypeError("Please provide dictionary of parameters \
                    to edit")

        #recompute waveform and fft
        self.h_plus, self.h_cross, self.t, \
                self.power, self.freqs = compute_fft(self._params)
    

    def bin_spec(self): 
        """
        Create histogram of power/freq
        """

        #create bin edges
        bin_edges = np.linspace(self.bin_range[0], self.bin_range[1], \
                            self.bins+1)

        #as in drasco, power over total power
        normed_spec = self.power/sum(self.power)

        #downselect power vals based on cutoff
        downselect_idx = np.where(np.log10(normed_spec)\
                        > self.power_cutoff)

        #downselected(scaled) power and frequency
        downselected_power = (normed_spec)[downselect_idx]
        downselected_freqs = self.freqs[downselect_idx]

        print("Number of downselected values:", len(downselected_power))

        hist, bins = np.histogram(abs(downselected_freqs), bins=bin_edges, \
                                    weights=downselected_power)

        return hist.real, bins

