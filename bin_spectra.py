#Monica Rizzo, 2020
import numpy as np
from compute_wf_fft import compute_fft


class DiscretizeSpectra(object):
    """
    compute and bin spectrum of an EMRI waveform snapshot
    """

    def __init__(self, params, fname="example", use_cl=False, bins=None, \
            angle_avg=False, bin_range=[0.0, 0.014], power_cutoff=-5.):
        """
        params: dict of params handed to AAK solver
        bins: number of frequency bins (if binning)
        bin_range: range of histogram
        power_cutoff: exponent of cutoff power
        """

        #create private parameter attribute
        self._params = params
        
        self.bins = bins
        self.bin_range = bin_range
        self.power_cutoff = power_cutoff

        self.fname = fname
        self.use_cl = use_cl
        self.angle_avg = angle_avg
            

        #compute waveform and fft and store locally 
        self.h_plus, self.h_cross, self.t, \
                self.power, self.freqs = compute_fft(self._params, \
                fname=self.fname, use_cl=self.use_cl)

        
        if self.angle_avg:
            self.orientation_avg()



    def change_params(self, change_params, fname="example", use_cl=False):
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
                self.power, self.freqs = compute_fft(self._params, \
                fname=fname, use_cl=use_cl)
    
        if self.angle_avg:

            self.orientation_avg()

    def orientation_avg(self): 
        """
        Average waveform (idk about this actually) and spectrum
        over orientation
        """

        #fix this for now, maybe make variable later
        n_int = 100

        h_plus = np.zeros(len(self.h_plus))
        h_cross = np.zeros(len(self.h_cross))
        power = np.zeros(len(self.power), dtype=np.cdouble)

        for i in range(n_int):

            #only vary the angle of the bh spin
            #TODO: fix sampling to be in cos(orbital plane inclination)
            #or something comparable?
            #or just up the count I guess
            tk, pk = np.random.uniform(0, 2.* np.pi, 2)
            
            #reassign
            self._params['theta_K'] = tk
            self._params['phi_K'] = pk

            print("phi_K", pk)
            print("theta_K", tk)

            #recompute
            self.h_plus, self.h_cross, self.t, \
                self.power, self.freqs = compute_fft(self._params, \
                fname=self.fname, use_cl=self.use_cl)
            
            #add to cumulative
            h_plus+=self.h_plus
            h_cross+=self.h_cross
            power+=self.power

        #average
        self.h_plus = h_plus/n_int
        self.h_cross = h_cross/n_int
        self.power = power/n_int


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

