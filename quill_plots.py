import h5py
import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg as la

#for nicer plot formatting
#RUN THIS TWICE TO UPDATE PLOT FORMATTING
plt.rcParams.update({'font.size': 16})

import os
import numpy.linalg as la


class PlotGriddedSpectra(object):

    """
    Read gridded data and generate quill plots and 
    single spectum cross section plots
    """

    def __init__(self, fname):

        #get cwd, if not in data folder, append to path

        path = os.getcwd()

        self.data_path = path + "/data"
            
        #load data
        self.raw_data = h5py.File(self.data_path + "/" + fname, 'r')

        #grid parameters
        self.params = list(self.raw_data.keys())
        self.params.remove('frequencies')
        self.params.remove('power')

        #remove timeseries data too if stored
        if "h_plus" in self.params:
            self.params.remove("h_plus")
            self.params.remove("h_cross")
            self.params.remove("times")

        #get range of unique parameter value
        for param in self.params:
           
            exec("self.unique_{} = np.unique(self.raw_data['{}'][()])".format(param, param))
       
    @property
    def parameter_ranges(self):

        out = {}

        for param in self.params:
            
            exec("out['{}'] = self.unique_{}".format(param, param))

        return out


    def plot_single_waveform(self, param_set):
        """
        takes param set dict specifying index of value on grid, or value
        **maybe edit this to interpolate between values?**
        ex: param_set = {'e':5, 'i': 1}
            param_set = {'e', 0.2, 'i': 15.0}
        """


        #check to make sure waveform data exists
        if "h_plus" not in self.raw_data.keys():

            raise KeyError("No waveform data to plot")
        
        loc_list = []

        #build location string

        if all(isinstance(v, int) for v in param_set.values()):

            for key in param_set.keys():
                val = eval("self.unique_{}[{}]".format(key, param_set[key]))
                str_val = '(self.raw_data["{}"][()] == {})'.format(key, val)
                loc_list.append(str_val)

        elif all(isinstance(v, float) for v in param_set.values()):

            for key in param_set.keys():
                str_val = '(self.raw_data["{}"][()] == {})'.format(key, param_set[key])
                loc_list.append(str_val)

        else:

            raise TypeError("Please provide a dictionary with either all integer values \
                    or all floats")


        loc_string = " & ".join(loc_list)

        index = eval("np.where({})".format(loc_string))

        #plot waveforms
        fig, axs = plt.subplots(2, figsize=(10, 6), sharex=True)
    
        #set title to fixed params
        title_list = []

        if all(isinstance(v, int) for v in param_set.values()):

            for key in param_set.keys():
                val = eval("self.unique_{}[{}]".format(key, param_set[key]))
                title_list.append("{} = {:.3f}".format(key, val))

        elif all(isinstance(v, float) for v in param_set.values()):

            for key in param_set.keys():
                title_list.append("{} = {:.3f}".format(key, param_set[key]))

        axs[0].set_title("GW Strain, {}".format(', '.join(title_list)))
        axs[0].plot(self.raw_data['times'][()], \
                self.raw_data['h_plus'].value[index][0])
        axs[0].set_ylabel("$h_+$")

        axs[1].plot(self.raw_data['times'][()], \
                self.raw_data['h_cross'].value[index][0])
        axs[1].set_ylabel("$h_x$")
        axs[1].set_xlabel("t")
        fig.tight_layout()
        
        return fig


    def plot_single_spectrum(self, param_set, logscale=True, xlim=[0.0, 0.014]):
        """
        takes param set dict specifying index of value on grid, or value
        **maybe edit this to interpolate between values?**
        ex: param_set = {'e':5, 'i': 1}
            param_set = {'e', 0.2, 'i': 15.0}
        """

        loc_list = []

        #build location string

        if all(isinstance(v, int) for v in param_set.values()):
            
            for key in param_set.keys():
                val = eval("self.unique_{}[{}]".format(key, param_set[key]))
                str_val = '(self.raw_data["{}"][()] == {})'.format(key, val)
                loc_list.append(str_val)

        elif all(isinstance(v, float) for v in param_set.values()):

            for key in param_set.keys():
                str_val = '(self.raw_data["{}"][()] == {})'.format(key, param_set[key])
                loc_list.append(str_val)

        else: 

            raise TypeError("Please provide a dictionary with either all integer values \
                    or all floats")


        loc_string = " & ".join(loc_list)
    
        index = eval("np.where({})".format(loc_string))

        fig = plt.figure(figsize=(10, 6))
       
        #set title to fixed params
        title_list = []

        if all(isinstance(v, int) for v in param_set.values()):

            for key in param_set.keys():
                val = eval("self.unique_{}[{}]".format(key, param_set[key]))
                title_list.append("{} = {:.3f}".format(key, val))

        elif all(isinstance(v, float) for v in param_set.values()):

            for key in param_set.keys():
                title_list.append("{} = {:.3f}".format(key, param_set[key]))

        plt.title("Power Spectrum, {}".format(', '.join(title_list)))

        if logscale:
            plt.semilogy(self.raw_data['frequencies'][()], \
            self.raw_data['power'].value[index][0] / \
            sum(self.raw_data['power'].value[index][0]), \
            marker='o', linestyle='None', markersize=3)
        else:
            plt.plot(self.raw_data['frequencies'][()], \
            self.raw_data['power'].value[index][0] / \
            sum(self.raw_data['power'].value[index][0]), \
            marker='o', linestyle='None', markersize=3)

        plt.xlim(xlim[0], xlim[1])
        plt.ylim(1e-5, 1)


        return fig

    def quill_plot(self, fixed_params, xlim=[0.0, 0.014]):
        """
        fixed_params: dict with name of fixed params and either values
        or indices of param values
        """

        #which parameter is the grid parameter
        l = [p not in fixed_params.keys() for p in self.params]
        grid_param = np.array(self.params)[l][0]
    
        #get positive frequencies and create grid of appropriate size
        f_pos = np.where((self.raw_data['frequencies'][()] > 0) \
                & (self.raw_data['frequencies'][()] < 0.014))[0]
        n_freq = len(f_pos)
        n_grid = eval("len(self.unique_{})".format(grid_param))

        dat = np.zeros((n_grid, n_freq))

        loc_list = []

        #build location string

        if all(isinstance(v, int) for v in fixed_params.values()):

            for key in fixed_params.keys():
                val = eval("self.unique_{}[{}]".format(key, fixed_params[key]))
                str_val = '(self.raw_data["{}"][()] == {})'.format(key, val)
                loc_list.append(str_val)

        elif all(isinstance(v, float) for v in fixed_params.values()):

            for key in fixed_params.keys():
                str_val = '(self.raw_data["{}"][()] == {})'.format(key, fixed_params[key])
                loc_list.append(str_val)

        else:

            raise TypeError("Please provide a dictionary with either all integer values or \
                    all floats")


        loc_string = " & ".join(loc_list)

        indices = eval("np.where({})[0]".format(loc_string))


        for i, j in zip(indices, range(n_grid)):
        
            vals = np.log10(self.raw_data['power'][()][i][f_pos]\
                    /sum(self.raw_data['power'][()][i][f_pos]))
                   
            dat[j, :] = vals
        
        
        #re-map z and t vals
        x_tick_locs = np.linspace(0, n_freq-1, 6, dtype=int)
        x_tick_lbls = ["%.3f" % i for i in self.raw_data['frequencies'].value[f_pos][x_tick_locs]]

        #prevent overcrowding axes
        if n_grid > 10:
            y_tick_locs = np.linspace(0, n_grid-1, 10, dtype=int)
        else:
            y_tick_locs = np.array(range(n_grid))
    
        y_tick_lbls = eval("['%.3f' % i for i in self.raw_data['{}'].value[indices][y_tick_locs]]".format(grid_param))
        y_tick_lbls.reverse()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(dat,  extent=[0, n_freq, 0, n_grid], aspect='auto')
        ax.set_xlabel('xlabel')


        #set title to fixed params
        title_list = []
       
        if all(isinstance(v, int) for v in fixed_params.values()):

            for key in fixed_params.keys():
                val = eval("self.unique_{}[{}]".format(key, fixed_params[key]))
                title_list.append("{} = {:.3f}".format(key, val))

        elif all(isinstance(v, float) for v in fixed_params.values()):

            for key in fixed_params.keys():
                title_list.append("{} = {:.3f}".format(key, fixed_params[key]))


        ax.set_title(", ".join(title_list))
           
        cbar = plt.colorbar(im)
        cbar.set_label("log($\\frac{dE}{dt}$ / $\\frac{dE}{dt}_{\mathrm{total}}$)")
        plt.xticks(x_tick_locs, x_tick_lbls)
        plt.yticks(y_tick_locs, y_tick_lbls)
        plt.xlabel("$f$ (Hz)")
        plt.ylabel(grid_param)

        return fig        
            
