import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#originally 20, 20, 20
def plotting_stuff(ax, title, xlabel, ylabel, legend = False, grid = False, ticklabel = 15, axislabel = 15, titlelabel = 15):
    ax.set_title(title, fontsize = titlelabel)
    ax.set_ylabel(ylabel, fontsize = axislabel)
    ax.set_xlabel(xlabel, fontsize = axislabel)
    ax.tick_params(labelsize = ticklabel)
    if(grid):
        ax.grid()
    if(legend):
        ax.legend(fontsize = axislabel-5)
        
#originally 8x8
class Plotter:
    def __init__(self, title, xlabel, ylabel, save_name = "", legend = False, xlen = 5, ylen = 5):
        self.title = title
        self.xlabel = xlabel 
        self.ylabel = ylabel
        self.legend = legend

        self.save_path = "."
        self.save_name = save_name
        
        self.fig = plt.figure(figsize = (xlen,ylen)) 
        self.ax = plt.subplot(111)
      
        plotting_stuff(self.ax, self.title, self.xlabel, self.ylabel, self.legend)
    
    def show_legend(self):
        #sloppy
        self.ax.legend(fontsize = 15)
        
    def savefig(self):
        path = "/home/lqcd/brian137/chroma_Wloops/gpu_test/analysis/Figures"
        self.fig.savefig(f"{path}/{self.save_name}", bbox_inches='tight')