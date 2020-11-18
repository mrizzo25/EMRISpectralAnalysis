#Monica Rizzo, 2020
#Utility functions for running AAK waveform generation
#from commandline

import numpy as np
import os



def parse_wf_opts(param_dict, fname, trajectory="true", \
                  snr="true", timing="true"):
    """
    Take python interface parameter dict and translate to 
    ascii file
    
    param_dict: dictionary of aak parameters
    fname: output file name
    trajectory: output trajectory or not ("true" or "false")
    snr: output snr or not ("true" or "false")
    timing: output timing information ("true" or "false")

    *extra options will work with boolean values too
    """

    #check if wf_data folder exists and create if not
    if "wf_data" not in os.listdir():
        os.mkdir("wf_data")

    #put parameters in list 
    param_list = list(param_dict.values())

    #insert extra values
    param_list.insert(0, "wf_data/"+fname)
    param_list.insert(3, trajectory)
    param_list.insert(4, snr)
    param_list.insert(5, timing)

    #convert all python booleans to strings
    for i in range(len(param_list)):
        if type(param_list[i]) is bool:
            param_list[i] = str(param_list[i]).lower()

    #write settings file
    with open("wf_data/"+fname+"_settings",'w') as f:
        for val in param_list:
            f.write('{}\n'.format(val))

def run_cl_wf_generation(fname):
    """
    Generate waveform, save, load, and return
    """

    #load path to emri kludge suite
    eks_path = os.environ["EKS_PATH"]

    #run waveform generation
    os.system("{}/bin/AAK_Waveform wf_data/{}_settings".format(eks_path,\
             fname))

    #load waveform data
    wf_dat = np.loadtxt("wf_data/{}_wave.dat".format(fname))

    t, h_plus, h_cross = wf_dat.T 

    return t, h_plus, h_cross
