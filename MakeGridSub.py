import numpy as np
import argparse
import h5py
import os
import shutil

from parse_ini import read_ini

class GridGen(object):
	"""
	Generate a grid of EMRI parameters for job scheduler to run
	"""

	def __init__(self, args):
		
		#output file name and ini file name
		self.fname = args.fname
		self.default = args.default

		#grid specifications
		self.grid_param = args.grid_param
		self.n_grid = args.n_grid
		self.log_scale_param = args.log_scale_param

		#local instances of params + grid ranges		
		self.param_grid_ranges = self.__default_grid_ranges()
		self.params = read_ini(self.ini)

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


		self.p_isco_fraction = args.p_isco_fraction
		self.p_mb_fraction = args.p_mb_fraction
		self.fix_sma = args.fix_sma
	
		#explicitly cast as floats just in case 
		if self.p_isco_fraction is not None:
            self.p_isco_fraction = float(self.p_isco_fraction)

        if self.p_mb_fraction is not None:
            self.p_mb_fraction = float(self.p_mb_fraction)

        if self.fix_sma is not None:
            self.fix_sma = float(self.fix_sma)	

		self.overwrite = args.overwrite
		self.verbose = args.verbose

		
		#create data dir if not already extant
		if fname+"_submit" in os.listdir("./"):

			if self.overwrite:
                shutil.rmtree(fname+'_submit')
				os.mkdir(fname+"_submit")			
	
            else:
                raise FileExistsError("Please pick different file name, \
                            or enable overwrite")
        else:
            os.mkdir(fname+"_submit")



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

		
		self.make_grid()
		self.write_grid()


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
                    'fixsma': [8.0, 12.0], 
					'p': [6.0, 10.0]}

        return param_grid_ranges

	def fractional_p_isco(self, param_header, param_set):

		#if 's' not a grid parameter, use the static value
        if 's' in param_header:
            
			s = param_set[:, param_header.index('s')]
        
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
        if 'piscofrac' in param_header:

            p_vals = param_set[:, param_header.index('piscofrac')] * p_isco
		
        #or else use the fixed value
        else:

			p_vals = self.p_isco_fraction * p_isco
                  

		#add p to parameter set and header list
		param_set = np.hstack((param_set, np.reshape(p_vals, (len(p_vals, 1)))))

		param_header.append('p')
 		
        return param_header, param_set


	def fractional_p_mb(self, param_header, param_set):

        #if 's' not a grid parameter, use the static value
        if 's' in param_header:
            
			s = param_set[:, param_header.index('s')]
        
		else:
            
			s = self.params['s']

        p_mb = 2 - s + 2 * (1 - s)**0.5

        if 'pmbfrac' in param_header:
			
			p_vals = param_set[:, param_header.index('pmbfrac')] * p_mb

        #or else use the fixed value
        else:
            
			p_vals = self.p_mb_fraction * p_mb

		#add p to parameter set and header list
        param_set = np.hstack((param_set, np.reshape(p_vals, (len(p_vals, 1)))))

        param_header.append('p')

        return param_header, param_set

	def fix_semimajor_axis(self, param_header, param_set):
        
		if 'e' in param_header:

            e = param_set[:, param_header.index('e')]

        else:
            
			e = self.params['e']

        if 'fixsma' in param_header:
	
			p_vals = param_set[:, param_header.index('fixsma')] * (1. - e**2)

        else:

			p_vals = self.fix_sma * (1. - e**2)

		#add p to parameter set and header list
        param_set = np.hstack((param_set, np.reshape(p_vals, (len(p_vals, 1)))))

        param_header.append('p')

		return param_header, param_set


	def write_grid(self):
	
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



        exec("{} = np.meshgrid({})"\
            .format(', '.join([p.upper() for p in self.grid_param]), \
                    ', '.join([p+'_vals' for p in self.grid_param])))

        #create an array of parameter set values at every grid point
        exec("self.param_set = np.array(list(map(list, zip({}))))"\
            .format(', '.join([p.upper()+'.ravel()' for p in self.grid_param])))

		param_header = list(self.grid_param)


		if self.p_isco_fraction is not None or 'piscofrac' in self.grid_param:

            param_header, param_set = self.fractional_p_isco(param_header, param_set)

        elif self.p_mb_fraction is not None or 'pmbfrac' in self.grid_param:

            param_header, param_set = self.fractional_p_mb(param_header, param_set)

        elif self.fix_sma is not None of 'fixsma' in self.grid_param:

            param_header, param_set = self.fix_semimajor_axis(param_header, param_set)

		#save or overwrite
		if self.fname+"_grid.dat" in os.listdir("data"):
			if self.overwrite:
				np.savetxt(fname+"_submit/"+fname+"_grid.dat", param_set, " ".join(param_header))
			else:
				raise FileExistsError("Please pick different file name, \
                            or enable overwrite")
		else:
			np.savetxt(fname+"_submit/"+fname+"_grid.dat", param_set, " ".join(param_header))


	def write_sub(self):
	

		#make data folder for script to write to
		os.mkdir(fname+"_submit/data")

		out_path = os.get_cwd() + "/" + fname + "_submit/data"


		#write slurm options at top of script
		#load modules and source env

		substring = "sbatch"
	
		#format options that get passed to single wf	
	


#########Script#########	

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
	
parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help='filename of output')
parser.add_argument("--ini", type=str, help='file with default parameters')
parser.add_argument("--grid-param", type=str, action="append", help="param to grid over")
parser.add_argument("--n-grid", nargs="+", type=int, help="number of points for each param on grid")

parser.add_argument("--log-scale-param", type=str, action="append", help="use log spacing for this params on grid")

parser.add_argument("--p-isco-fraction", default=None, help="fix slr to fractional value of isco slr")
parser.add_argument("--p-mb-fraction", default=None, help="fix value of marginally bound radius")
parser.add_argument("--fix-sma", default=None, help="fix semimajor axis value")

parser.add_argument("--overwrite", action='store_true', help="if output data file already exists, overwrite it")
parser.add_argument("--verbose", action='store_true', help="print extra debugging output")

parser.add_argument("--slurm-opts", type=str, help="string containing slurm options. format as 'option1=val option2=val ...'")
parser.add_argument("--slurm-module", type=str, action="append", help="one or more modules to load in slurm scripts")
parser.add_argument("--slurm-env", type=str, default=None, help="environment to source for slurm scripts")

parsed, unknown = parser.parse_known_args()

#accept additional params as emri params
for arg in unknown:

	#if passed a parameter grid code recognizes, make a valid argument
    if arg.startswith(("--", "-")) and (arg.replace('--', '') in param_type_dict.keys() or \
            arg.replace('-', '') in param_type_dict.keys()):

        if arg.startswith("--"):
            key = arg.replace('--', '')
        else:
            key = arg.replace('-', '')

        parser.add_argument(arg, type=param_type_dict[key])

    #accept params of the form "param-max" and "param-min"
	#to define grid ranges
    elif arg.endswith("max") and arg.replace('--', '').split('-')[0] in param_type_dict.keys():

        parser.add_argument(arg, type=float)

    elif arg.endswith("min") and arg.replace('--', '').split('-')[0] in param_type_dict.keys():

        parser.add_argument(arg, type=float)

args = parser.parse_args()

#run grid generation module
GridGen(args)


