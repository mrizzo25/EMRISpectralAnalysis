#Monica Rizzo, 2020
import numpy as np
import AAKwrapper

from scipy import fftpack
from scipy import signal

from interface_w_cl import *


def compute_fft(wf_params, use_cl=False, fname="example"):
    """
    For a set of default parameters, compute the 
    waveform and fft of a snapshot

    MAKE SURE LISA RESPONSE IS SET TO FALSE
    """

    #this should be a dictionary
    #make sure the keys match up

    if use_cl:
        #generate wf from command line
        parse_wf_opts(wf_params, fname)
        t, h_plus, h_cross = run_cl_wf_generation(fname)
    else:
        #generate waveform
        t, h_plus, h_cross, timing = AAKwrapper.wave(wf_params)

    #total signal as defined in Drasco
    h_total = h_plus - 1j * h_cross

    #make a window for the data to taper the edges
    window = signal.tukey(wf_params['length'], alpha=0.1, sym = False)

    tapered_h_total =  h_total * window

    #if this starts running slow, maybe re-itroduce zero padding
    #so far the fft has not been a problem

    h_of_f = fftpack.fft(tapered_h_total)

    freqs = fftpack.fftfreq(len(tapered_h_total), d=wf_params['dt'])

    #total power
    power = np.sqrt((h_of_f.real + 1j*h_of_f.imag) * \
                    (h_of_f.real - 1j*h_of_f.imag))

    return h_plus, h_cross, t, power, freqs



