from neuron import h

class Synapses:
    def __init__(self,source,target,section,nrel = 0,syntype = 'ANK',record_all = 0,weight = 1):
        self.record_all = record_all
        if record_all:
            print("Recording all in Synapse")

        self.input = h.NetStim(0.5)
        self.input.start = -10
        self.input.number = 1
        self.input.interval = 1e9
        self.weight = weight

        self.nrel = nrel
        self.syntype = syntype

        self.postsyns = {}

        if (type(source) == type('s')):
            sourcetype = source
        else:
            sourcetype = source.whatami

        if self.record_all:
            self.SpikeTrain_input = [h.Vector(),h.Vector()]
            self.netcon_in = h.NetCon(self.input,None, 0, 0.1, 1)
            self.netcon_in.record(self.SpikeTrain_input[1], self.SpikeTrain_input[0], 1)
           
                


#basket
        if sourcetype == 'pf':
            if target.whatami == 'bc':
                self.postsyns['AMPA'] = [h.Golgi_PF_syn(0.5, sec=section)]
                self.postsyns['AMPA'][0].tau_facil = 10.8
                self.postsyns['AMPA'][0].tau_rec = 35.1
                self.postsyns['AMPA'][0].tau_1 = 10
                self.postsyns['AMPA'][0].gmax = 1200
                self.postsyns['AMPA'][0].U = 0.13
                
                self.nc_syn = [h.NetCon(self.input,receptor[0],0,0.1,1) for receptor in self.postsyns.values()]
    
        elif sourcetype == 'pfnmda':
            if target.whatami == 'bc':
                self.postsyns['NMDA'] = [h.SC_NMDA_NR2B(0.5, sec=section)] #, sec=section)] #location 0.3
                self.postsyns['NMDA'][0].tau_facil = 10.8
                self.postsyns['NMDA'][0].tau_rec = 35.1
                self.postsyns['NMDA'][0].tau_1 = 10
                self.postsyns['NMDA'][0].gmax = 5000
                self.postsyns['NMDA'][0].U = 0.13

                self.nc_syn = [h.NetCon(self.input,receptor[0],0,0.1,1) for receptor in self.postsyns.values()]
                

        elif sourcetype == 'gaba_bc':
            if target.whatami == 'bc':
                self.postsyns['GABA'] = [h.PC_gaba_alpha1(0.5)] # self.postsyns sec=section
                self.postsyns['GABA'][0].tau_facil =  4
                self.postsyns['GABA'][0].tau_rec = 15
                self.postsyns['GABA'][0].tau_1 = 1
                self.postsyns['GABA'][0].Erev = -70
                self.postsyns['GABA'][0].gmaxA1 = 1600
                self.postsyns['GABA'][0].U = 0.42
                
                self.nc_syn = [h.NetCon(self.input,receptor[0],0,0.1,1) for receptor in self.postsyns.values()]

        
        else:
            print('SOURCE TYPE DOES NOT EXIST SOMETHING WRONG!!!!!!!!!')
            
            

        if len(self.postsyns) > 0:
            self.i = {}
            for (post_type,post) in self.postsyns.items():
                for p in post:
                    self.i[post_type] = []
                    self.i[post_type].append(h.Vector())
                    self.i[post_type][-1].record(p._ref_i)