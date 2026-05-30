from neuron import h
import math
import numpy as np
from Synapses import Synapses
import random

class Basket:
    def __init__(self,record_all = 1):
        h.load_file('stdlib.hoc')
        h.load_file('import3d.hoc')
        cell = h.Import3d_Neurolucida3()
        cell.input('morphology/01bc.ASC')  
        i3d = h.Import3d_GUI(cell, 0)
        i3d.instantiate(self)
        
        self.soma[0].nseg = 1 + (2*int(self.soma[0].L/40))
        self.soma[0].Ra = 122
        self.soma[0].cm = 1 
        
        self.soma[0].insert('Leak')
        self.soma[0].gmax_Leak = 0.00004
        self.soma[0].e_Leak = -55
        
        self.soma[0].insert('Nav1_1')
        self.soma[0].gbar_Nav1_1 = 0.2 
        self.soma[0].ena = 60  
        
        self.soma[0].insert('Cav3_2')
        self.soma[0].gcabar_Cav3_2 = 0.0001
        
        self.soma[0].insert('Cav12')
        self.soma[0].gbar_Cav12 = 0.0007 
        
        self.soma[0].insert('Cav13')
        self.soma[0].gbar_Cav13 = 0.000005 
        
        self.soma[0].insert('Kir2_3')
        self.soma[0].gkbar_Kir2_3 = 0.0001 
        
        self.soma[0].insert('Kv3_4')
        self.soma[0].gkbar_Kv3_4 = 0.097 
        
        self.soma[0].insert('Kv4_3')
        self.soma[0].gkbar_Kv4_3 = 0.01 
        
        
        self.soma[0].insert('Kca3_1')
        self.soma[0].gkbar_Kca3_1 = 0.001 
        self.soma[0].ek = -80
        
        self.soma[0].insert('HCN1_PC')
        self.soma[0].gbar_HCN1_PC = 0.001 
        self.soma[0].eh = -34 
        
        self.soma[0].insert('cdp5StCmod')
        self.soma[0].TotalPump_cdp5StCmod = 2e-9
        
        self.soma[0].push()
        
        self.soma[0].eca = 137.5
        
        h.pop_section()
	
        self.whatami = "bc"
        
        for i in self.dend:  
            
            i.nseg = 1 + 2*int(i.L/40)
         
            i.Ra = 122
            i.cm = 1
            
            i.insert('Leak')
            i.gmax_Leak =  0.00001 
            i.e_Leak = -55 
            
            i.insert('Cav3_2')
            i.gcabar_Cav3_2 = 0.00005 
        
            i.insert('Cav12')
            i.gbar_Cav12 = 0.0002 
        
            i.insert('Cav13')
            i.gbar_Cav13 = 0.000005 
            
            i.insert('Kv4_3')
            i.gkbar_Kv4_3 = 0.00987201764943
            
            i.insert('Kca2_2')
            i.gkbar_Kca2_2 = 0.0065 
            i.ek = -80
            
            i.insert('cdp5StCmod')
            
            i.TotalPump_cdp5StCmod = 2e-9 
	    
            i.push()
            i.eca = 137.5
        
            h.pop_section()
	    
            
        for i, d in enumerate(self.axon):
            if i == 0:
                #print('ais:', i)
            
                self.axon[i].nseg = 1 + 2*int(self.axon[i].L/40)
                self.axon[i].Ra = 122
                self.axon[i].cm = 1
                
                self.axon[i].insert('Leak')
                self.axon[i].gmax_Leak = 0.00001 
                self.axon[i].e_Leak = -55 
                
                self.axon[i].insert('Nav1_6')
                self.axon[i].gbar_Nav1_6 = 0.3 
                self.axon[i].ena = 60
                
                self.axon[i].insert('Kv3_4')
                self.axon[i].gkbar_Kv3_4 = 0.002 
                self.axon[i].ek = -80  

                
                self.axon[i].insert('HCN1_PC')
                self.axon[i].gbar_HCN1_PC = 0.001 
                self.axon[i].eh = -34  
                
                self.axon[i].insert('Kca1_1')
                self.axon[i].gbar_Kca1_1 = 0.01 
                        
                self.axon[i].insert('Cav2_1')
                self.axon[i].pcabar_Cav2_1 = 2.2e-4 
                
                self.axon[i].insert('cdp5StCmod')
                
                self.axon[i].TotalPump_cdp5StCmod = 2e-9 
                        
                self.axon[i].push()
                        
                self.axon[i].eca = 137.5

                h.pop_section()
                
                
            else:
                
                self.axon[i].nseg = 1 + 2*int(self.axon[i].L/40)
            
                self.axon[i].Ra = 122
                self.axon[i].cm = 1
                
                self.axon[i].insert('Leak')
                self.axon[i].gmax_Leak = 0.000001 
                self.axon[i].e_Leak = -55 
                
                self.axon[i].insert('Nav1_6')
                self.axon[i].gbar_Nav1_6 = 0.001 
                self.axon[i].ena = 60
            
                self.axon[i].insert('Kv3_4')
                self.axon[i].gkbar_Kv3_4 = 0.001
                self.axon[i].ek = -80  
            
                self.axon[i].insert('Kv1_1')
                self.axon[i].gbar_Kv1_1 = 0.0005 
            
                self.axon[i].insert('HCN1_PC')
                self.axon[i].gbar_HCN1_PC = 0.0001 
                self.axon[i].eh = -34  
                
                self.axon[i].insert('Kca1_1')
                self.axon[i].gbar_Kca1_1 = 0.001 
                    
                self.axon[i].insert('Cav2_1')
                self.axon[i].pcabar_Cav2_1 = 0.00008 

                self.axon[i].insert('cdp5StCmod')
                
                self.axon[i].TotalPump_cdp5StCmod = 2e-9
                    
                self.axon[i].push()
                    
                self.axon[i].eca = 137.5
                
                h.pop_section()
        
        self.voltage = h.Vector()
        self.voltage.record(self.soma[0](0.5)._ref_v)

        self.time = h.Vector()
        self.time.record(h._ref_t)

	##Synapses
    def createsyn(self, npf): #temporaneo
        
        self.pf_bc = [] 
        self.pf_bcnmda = []
        self.bc_bc = []
        
        self.pf_bc_nmda_axon = []

        
        
        self.new_dend = []
        
        for i in self.dend:
            if i.L <= 60:
                self.new_dend.append(i)
         
        n = len(self.new_dend)
        
        self.pfrand = []
        
        for j in range(npf):
            n -= 1
            i = random.randint(0, n)	    
            self.new_dend[i], self.new_dend[n] = self.new_dend[n], self.new_dend[i]
            self.pfrand.append(self.new_dend[n])
            self.pf_bc.append(Synapses('pf',self,self.pfrand[j]))   
            self.pf_bcnmda.append(Synapses('pfnmda',self,self.pfrand[j]))
        
       
        self.dendinhib = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21]
        
        for e in self.dendinhib:
            self.bc_bc.append(Synapses('gaba_bc',self,self.dend[e]))
        print('BC_BC', len(self.bc_bc))