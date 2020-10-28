import numpy as np
import os



def parse_wf_opts(param_dict, fname, trajectory="true", \
                  snr="true", timing="true"):


    #check if wf_data folder exists and create if not
    if "wf_data" not in os.listdir():
        os.mkdir("wf_data")

    #put parameters in list 
    param_list = list(param_dict.values())

    #insert extra values
    param_list.insert(0, "wf_data"+fname)
    param_list.insert(3, trajectory)
    param_list.insert(4, snr)
    param_list.insert(5, timing)


    #write settings file
    with open("wf_data/"+fname+"_settings",'w') as f:
        for val in param_list:
            f.write('{}\n'.format(val))

def run_cl_wf_generation(fname):
    """
    Generate waveform, save, load, and return
    """

    eks_path = os.environ["EKS_PATH"]

    os.system("{}/bin/AAK_Waveform wf_data/{}_settings".format(eks_path,\
             fname))

    wf_dat = np.loadtxt("wf_data/{}_wave.dat".format(fname))

    t, h_plus, h_cross = wf_dat.T 

    return t, h_plus, h_cross
