import xml.etree.ElementTree as ET 
import numpy as np
import matplotlib.pyplot as plt
import errors


def string_to_double_array(string):
    """Converts a string of doubles to an array of doubles."""
    return [float(x) for x in string.split()]

def extract_loops(data_path, file_number, start_time, which_loop):
    xmlfile = f"{data_path}/{file_number}.dat" 
    tree = ET.parse(xmlfile)  
    root = tree.getroot() 
    inline_observables = root.find("InlineObservables")
    elem = inline_observables.findall("elem")

    #Wloops of timesize t0 and t1 over all r for the file being read
    #has size num smoothings * r
    all_t0 = []
    all_t1 = []
    
    which_time = 0
    for item in elem:
        wloop = item.find("WilsonLoop")
        if wloop != None:    
            individual_t0 = [] #has size r
            individual_t1 = [] #has size r
            
            r_wilsloop2 = wloop.findall(f"wils_loop{which_loop}/wloop{which_loop}/elem/r")
            loop_wilsloop2 = wloop.findall(f"wils_loop{which_loop}/wloop{which_loop}/elem/loop")
            
            potential = []
            r = []
            for i, loops in enumerate(loop_wilsloop2):
                text_data = string_to_double_array(loops.text)
                individual_t0.append(text_data[start_time[which_time]])
                individual_t1.append(text_data[start_time[which_time]+1])
                r.append(int(r_wilsloop2[i].text))
        
            all_t0.append(individual_t0)
            all_t1.append(individual_t1)
            
            which_time = which_time + 1
 
    if which_loop == 3:
        chunk = int(len(r)/5)
        r = np.hstack((np.linspace(1, chunk*2, chunk*2)*2**0.5, 
                      np.linspace(1, chunk, chunk)*5**0.5, 
                      np.linspace(1, chunk*2, chunk*2)*3**0.5))
        
    return np.array(r), all_t0, all_t1


class Data_Processing:
    def __init__(self, data_path, start_time = [5,5,5,5]):
        self.data_path = data_path
        
        #will have size num files * num smoothings * r
        self.wloop_t0 = []
        self.wloop_t1 = []
        
        self.start_time = start_time
    
        self.file0 = 1250
        self.file1 = 2400
        self.file_step = 50
        self.off_axis = True
        
        self.n_bootstraps = 100
        self.els_per_bootstrap = 100
        
    def get_potential(self, number):
        r_loop2, all_t0_loop2, all_t1_loop2 = extract_loops(self.data_path, number, self.start_time, 2)
        if self.off_axis:
            r_loop3, all_t0_loop3, all_t1_loop3 = extract_loops(self.data_path, number, self.start_time, 3)
            self.wloop_t0.append(np.concatenate((all_t0_loop2, all_t0_loop3), axis = 1))
            self.wloop_t1.append(np.concatenate((all_t1_loop2, all_t1_loop3), axis = 1))
            self.r = np.concatenate((r_loop2, r_loop3)) 
            
        else:  
            self.wloop_t0.append(all_t0_loop2)
            self.wloop_t1.append(all_t1_loop2)
            self.r = np.array(r_loop2)
    
    def read_all_files(self):
        for i in range(self.file0, self.file1, self.file_step):
            self.get_potential(i)
    
        #sort it
        self.wloop_t0 = np.array(self.wloop_t0)
        self.wloop_t1 = np.array(self.wloop_t1)
        for i in range(self.wloop_t0.shape[0]):
            for j in range(self.wloop_t0.shape[1]):
                self.wloop_t0[i][j] = [x for _,x in sorted(zip(self.r, self.wloop_t0[i][j]))]
                self.wloop_t1[i][j] = [x for _,x in sorted(zip(self.r, self.wloop_t1[i][j]))]
        self.r = np.array(sorted(self.r))
        
    def rand_bootstraps(self):
        self.rand_indices = np.array([np.random.randint(0, self.wloop_t0.shape[0], self.els_per_bootstrap) 
                                      for i in range(self.n_bootstraps)])
        
    def construct_bootstrap(self, wloop0, wloop1):
        self.rand_bootstraps()
        dt_0 = np.average(wloop0[self.rand_indices], axis = 0)
        dt_1 = np.average(wloop1[self.rand_indices], axis = 0)
        return dt_0, dt_1
    
    def find_potential(self, wloop0, wloop1):
        dt_0, dt_1 = self.construct_bootstrap(wloop0, wloop1)
        N = len(dt_0)
        ratio = np.ma.masked_invalid(np.log(abs(dt_0 / dt_1))) #this abs is bad...
        mean = np.average(ratio, axis = 0)
        error = np.sqrt(np.sum((ratio - mean)**2, axis = 0)/N)      
        return [mean, error]

    def find_force(self, wloop0, wloop1):
        dt_0, dt_1 = self.construct_bootstrap(wloop0, wloop1)
        N = len(dt_0)
        ratio = np.ma.masked_invalid(np.log(abs(dt_0 / dt_1))) #this abs is bad...
        force = []
        for i in range(ratio.shape[0]):
            force.append(-ratio[i][1:] + ratio[i][-1])
        mean = np.average(force, axis = 0)
        error = np.sqrt(np.sum((force - mean)**2, axis = 0)/N)
        return [mean, error]
    
    def find_potential_errors(self):
        return np.array([self.find_potential(self.wloop_t0[:,i], self.wloop_t1[:,i]) for i in range(self.wloop_t0.shape[1])])
   
    def find_force_errors(self):
        return np.array([self.find_force(self.wloop_t0[:,i], self.wloop_t1[:,i]) for i in range(self.wloop_t0.shape[1])])
    
    
    
class jackknife_Data_Processing:
    def __init__(self, data_path, start_time = 5):
        self.data_path = data_path
        
        #will have size num files * num smoothings * r
        self.wloop_t0 = []
        self.wloop_t1 = []
        
        self.start_time = start_time
    
        self.file0 = 1250
        self.file1 = 2400
        self.file_step = 50
        self.off_axis = True
        
    def get_potential(self, number):
        r_loop2, all_t0_loop2, all_t1_loop2 = extract_loops(self.data_path, number, self.start_time, 2)
        if self.off_axis:
            r_loop3, all_t0_loop3, all_t1_loop3 = extract_loops(self.data_path, number, self.start_time, 3)
            self.wloop_t0.append(np.concatenate((all_t0_loop2, all_t0_loop3), axis = 1))
            self.wloop_t1.append(np.concatenate((all_t1_loop2, all_t1_loop3), axis = 1))
            self.r = np.concatenate((r_loop2, r_loop3)) 
            
        else:  
            self.wloop_t0.append(all_t0_loop2)
            self.wloop_t1.append(all_t1_loop2)
            self.r = np.array(r_loop2)
    
    def read_all_files(self):
        for i in range(self.file0, self.file1, self.file_step):
            self.get_potential(i)
    
        #sort it
        self.wloop_t0 = np.array(self.wloop_t0)
        self.wloop_t1 = np.array(self.wloop_t1)
        for i in range(self.wloop_t0.shape[0]):
            for j in range(self.wloop_t0.shape[1]):
                self.wloop_t0[i][j] = [x for _,x in sorted(zip(self.r, self.wloop_t0[i][j]))]
                self.wloop_t1[i][j] = [x for _,x in sorted(zip(self.r, self.wloop_t1[i][j]))]
        self.r = np.array(sorted(self.r))
        
    def make_potential_blocks(self, i, j):
        dt_0_blocks, N = errors.make_jackknife_blocks(self.wloop_t0[:,i][:,j])
        dt_1_blocks, N = errors.make_jackknife_blocks(self.wloop_t1[:,i][:,j])
        ratio = np.ma.masked_invalid(np.log10(abs(dt_0_blocks / dt_1_blocks))) 
        return ratio
    
    def make_force_blocks(self, i, j):
        r0_ratio = self.make_potential_blocks(i, j)
        r1_ratio = self.make_potential_blocks(i, j+1)
        force = -(r1_ratio - r0_ratio)
        return force

    def find_potential_errors(self):
        self.wloop_t0 = np.array(self.wloop_t0)
        self.wloop_t1 = np.array(self.wloop_t1)
        
        nsmooth_by_nr_mean = np.zeros((self.wloop_t0.shape[1],self.wloop_t0.shape[2]))
        nsmooth_by_nr_error = np.zeros((self.wloop_t0.shape[1],self.wloop_t0.shape[2]))
        
        for i in range(self.wloop_t0.shape[1]): 
            for j in range(self.wloop_t0.shape[2]): 
                mean, error = errors.find_error_blocks(self.make_potential_blocks(i, j))
                nsmooth_by_nr_mean[i][j] = mean
                nsmooth_by_nr_error[i][j] = error
                
        return nsmooth_by_nr_mean, nsmooth_by_nr_error
    
    def find_force_errors(self):
        '''
        bruh there's just a -1 that makes this not generalizable
        '''
        self.wloop_t0 = np.array(self.wloop_t0)
        self.wloop_t1 = np.array(self.wloop_t1)
        
        nsmooth_by_nr_mean = np.zeros((self.wloop_t0.shape[1],self.wloop_t0.shape[2]-1))
        nsmooth_by_nr_error = np.zeros((self.wloop_t0.shape[1],self.wloop_t0.shape[2]-1))
        
        for i in range(self.wloop_t0.shape[1]): 
            for j in range(self.wloop_t0.shape[2]-1): 
                mean, error = errors.find_error_blocks(self.make_force_blocks(i, j))
                nsmooth_by_nr_mean[i][j] = mean
                nsmooth_by_nr_error[i][j] = error
      
        return nsmooth_by_nr_mean, nsmooth_by_nr_error
    
    