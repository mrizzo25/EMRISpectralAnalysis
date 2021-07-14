import numpy as np


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
    'D': float}


def read_ini(fname):

	dict_out = {}

	with open(fname, 'r') as f:

		lines = f.readlines()

			for l in lines:
				
				#remove white space and newline char 				
				l.replace(" ", "")
				l.replace("\n", "")
				
				sl = l.split("=")

				if sl[0] in param_type_dict.keys():

				 	dict_out[sl[0]] = param_type_dict[sl[0]](sl[1])	

	return dict_out	
		


	

