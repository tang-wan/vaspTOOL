import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp1d, interp2d
import pyprocar as ppr

from STT_Tool import Tools
color = Tools.ColorList()
# >>>>>>>>>> <<<<<<<<<<
class Normal_vaspkitBANDplot():
    def __init__(self, B_FilePath, L_FilePath):
        # print("testing module...")
        self.B_FilePath = B_FilePath
        self.L_FilePath = L_FilePath

        data  = np.loadtxt(B_FilePath)
        Kpath = data[:,0]
        self.BANDdata = np.transpose(data[:,1:])

        L_label = np.loadtxt(L_FilePath, dtype='str', skiprows=1)
        labels = L_label[:,0]
        self.labels = np.where(labels == 'GAMMA', 'G', labels)

        ticks  = np.float64(L_label[:,1])

        normalf = np.max(Kpath)
        self.Kpath = np.round(Kpath/normalf, 4)
        self.ticks = np.round(ticks/normalf, 4)
# ==========
    def Read_AllData(self):
        LABELoutput = (self.labels, self.ticks)
        BANDoutput  = (self.Kpath, self.BANDdata)
        return BANDoutput, LABELoutput
# ==========    
    def Read_SpecKData(self, startK, endK, error=5e-4):
        labels   = self.labels
        ticks    = self.ticks
        Kpath    = self.Kpath
        BANDdata = self.BANDdata

        spec_Kath_start = np.where(np.abs(Kpath-startK) < error)[0]
        spec_Kath_start = np.min(spec_Kath_start)
        spec_Kath_end = np.where(np.abs(Kpath-endK) < error)[0]
        spec_Kath_end = np.max(spec_Kath_end)+1
        spec_Kpath    = Kpath[spec_Kath_start:spec_Kath_end]
        spec_BANDdata = BANDdata[:,spec_Kath_start:spec_Kath_end]

        spec_tick_start = np.where(np.abs(ticks-startK) < error)[0]
        spec_tick_start = np.min(spec_tick_start)
        spec_tick_end = np.where(np.abs(ticks-endK) < error)[0]
        spec_tick_end = np.max(spec_tick_end)+1
        spec_ticks  = ticks[spec_tick_start:spec_tick_end]
        spec_labels = labels[spec_tick_start:spec_tick_end]

        BANDoutput  = (spec_Kpath, spec_BANDdata)
        LABELoutput = (spec_labels, spec_ticks)

        return BANDoutput, LABELoutput, 
# ==========    
    def Edit_ResortData(self, BANDinput, LABELinput, sepK, error=5e-4):
        labels   = LABELinput[0]
        ticks    = LABELinput[1]
        Kpath    = BANDinput[0]
        BANDdata = BANDinput[1]

        Resort_ind = np.where(np.abs(ticks-sepK) < error)[0]
        Resort_ind = np.max(Resort_ind)
        
        K_shift = (ticks[-1]-ticks[0])
        Resort_labels = np.append(labels, labels[1:Resort_ind+1])
        Resort_ticks  = np.append(ticks, ticks[1:Resort_ind+1] + K_shift)

        Resort_ind = np.where(np.abs(Kpath-sepK) < error)[0]
        Resort_ind = np.max(Resort_ind)

        Resort_Kpath     = np.append(Kpath, Kpath[0:Resort_ind] + K_shift)
        Resort_BANDdata  = np.c_[BANDdata, BANDdata[:,0:Resort_ind]]
        
        BANDoutput  = (Resort_Kpath, Resort_BANDdata)
        LABELoutput = (Resort_labels, Resort_ticks)

        (self.Kpath, self.BANDdata) = (Resort_Kpath, Resort_BANDdata)
        (self.labels, self.ticks)   = (Resort_labels, Resort_ticks)

        return BANDoutput, LABELoutput, 
# ==========
class Project_vaspkitBANDplot():

    def __init__(self, NBAND, 
                 pB1_FilePath, pB2_FilePath,
                 pL_FilePath):
        self.NBAND = NBAND
        self.L_FilePath   = pL_FilePath
        self.pB1_FilePath = pB1_FilePath
        self.pB2_FilePath = pB2_FilePath
        
        pB1_data = np.loadtxt(pB1_FilePath)
        Kpath1 = pB1_data[:,0]
        Kpath1 = Kpath1.reshape(NBAND, len(Kpath1)//NBAND)
        pBNAD1_data = pB1_data[:,1:]
        pBNAD1_data = pBNAD1_data.reshape(NBAND, len(pBNAD1_data)//NBAND, 2)
        self.pBNAD1_data = pBNAD1_data

        pB2_data = np.loadtxt(pB2_FilePath)
        Kpath2 = pB2_data[:,0]
        Kpath2 = Kpath2.reshape(NBAND, len(Kpath2)//NBAND)
        pBNAD2_data = pB2_data[:,1:]
        pBNAD2_data = pBNAD2_data.reshape(NBAND, len(pBNAD2_data)//NBAND, 2)
        self.pBNAD2_data = pBNAD2_data

        L_label = np.loadtxt(pL_FilePath, dtype='str', skiprows=1)
        labels = L_label[:,0]
        ticks  = np.float64(L_label[:,1])
        self.labels = np.where(labels == 'GAMMA', 'G', labels)

        if np.max(Kpath1[0]) == np.max(Kpath2[0]):
            normalf = np.max(Kpath1)
            self.Kpath1 = np.round(Kpath1/normalf, 4)
            self.Kpath2 = np.round(Kpath2/normalf, 4)
            self.ticks  = np.round(ticks/normalf, 4)

        else:
            print("The Kpaths are not the same")
# ==========    
    def Read_AllData(self, p=True):
        LABELoutput  = (self.labels, self.ticks)
        BAND1output  = (self.Kpath1, self.pBNAD1_data)
        BAND2output  = (self.Kpath2, self.pBNAD2_data)
        if p:
            print("# ==========")
            print("The shape of band1")
            print(np.shape(self.Kpath1))
            print(np.shape(self.pBNAD1_data))
            print("# -----")
            print("The shape of band2")
            print(np.shape(self.Kpath2))
            print(np.shape(self.pBNAD2_data))
            print("# ==========")
        else:
            pass

        return BAND1output, BAND2output, LABELoutput
# ==========    
    def Read_SpecKData(self, startK, endK, error=5e-4):
        NBAND = self.NBAND

        labels   = self.labels
        ticks    = self.ticks
        
        Kpath1      = self.Kpath1
        pBNAD1_data = self.pBNAD1_data
        
        Kpath2      = self.Kpath2
        pBNAD2_data = self.pBNAD2_data

        spec_tick_start = np.where(np.abs(ticks-startK)<error)[0]
        spec_tick_start = np.min(spec_tick_start)
        spec_tick_end = np.where(np.abs(ticks-endK) < error)[0]

        spec_tick_end = np.max(spec_tick_end)
        new_ticks  = ticks[spec_tick_start:spec_tick_end+1]
        new_labels = labels[spec_tick_start:spec_tick_end+1]

        new_Kpath1 = np.array([])
        new_Kpath2 = np.array([])
        new_pBAND1 = np.array([])
        new_pBAND2 = np.array([])
        for ind in range(NBAND):
            spec_Kath1_start = np.where(np.abs(Kpath1[ind]-startK) < error)[0]
            spec_Kath1_start = np.min(spec_Kath1_start)
            spec_Kath1_end = np.where(np.abs(Kpath1[ind]-endK) < error)[0]
            spec_Kath1_end = np.max(spec_Kath1_end)
            
            if spec_Kath1_end < spec_Kath1_start:
                spec_Kath1_end, spec_Kath1_start = spec_Kath1_start+1, spec_Kath1_end

            spec_Kpath1 = Kpath1[ind][spec_Kath1_start:spec_Kath1_end]
            new_Kpath1  = np.append(new_Kpath1, spec_Kpath1)

            spec_pBAND1 = pBNAD1_data[ind][spec_Kath1_start:spec_Kath1_end]
            new_pBAND1  = np.append(new_pBAND1, spec_pBAND1)
            # ==========
            spec_Kath2_start = np.where(np.abs(Kpath2[ind]-startK) < error)[0]
            spec_Kath2_start = np.min(spec_Kath2_start)
            spec_Kath2_end = np.where(np.abs(Kpath2[ind]-endK) < error)[0]
            spec_Kath2_end = np.max(spec_Kath2_end)
            
            if spec_Kath2_end < spec_Kath2_start:
                spec_Kath2_end, spec_Kath2_start = spec_Kath2_start+1, spec_Kath2_end

            spec_Kpath2 = Kpath2[ind][spec_Kath2_start:spec_Kath2_end]
            new_Kpath2  = np.append(new_Kpath2, spec_Kpath2)

            spec_pBAND2 = pBNAD2_data[ind][spec_Kath2_start:spec_Kath2_end]
            new_pBAND2  = np.append(new_pBAND2, spec_pBAND2)

        new_Kpath1 = new_Kpath1.reshape(NBAND, len(new_Kpath1)//NBAND)
        new_pBAND1 = new_pBAND1.reshape(NBAND, len(new_pBAND1)//NBAND//2, 2)

        new_Kpath2 = new_Kpath2.reshape(NBAND, len(new_Kpath2)//NBAND)
        new_pBAND2 = new_pBAND2.reshape(NBAND, len(new_pBAND2)//NBAND//2, 2)
        
        BAND1output  = (new_Kpath1, new_pBAND1)
        BAND2output  = (new_Kpath2, new_pBAND2)
        LABELoutput  = (new_labels, new_ticks)
        return BAND1output, BAND2output, LABELoutput
# ==========    
    def Edit_ResortData(self, BANDinput1, BANDinput2, LABELinput, sepK, error=5e-4):
        NBAND = self.NBAND

        labels   = LABELinput[0]
        ticks    = LABELinput[1]
        
        Kpath1      = BANDinput1[0]
        pBNAD1_data = BANDinput1[1]
        
        Kpath2      = BANDinput2[0]
        pBNAD2_data = BANDinput2[1]
        # =====
        Resort_ind = np.where(np.abs(ticks-sepK) < error)[0]
        Resort_ind = np.max(Resort_ind)
        
        K_shift = (ticks[-1]-ticks[0])
        Resort_labels = np.append(labels, labels[1:Resort_ind+1])
        Resort_ticks  = np.append(ticks, ticks[1:Resort_ind+1] + K_shift)

        # =====
        new_Kpath1 = np.array([])
        new_pBAND1 = np.array([])
        new_Kpath2 = np.array([])
        new_pBAND2 = np.array([])
        for ind in range(NBAND):
            Resort_ind = np.where(np.abs(Kpath1[ind]-sepK) < error)[0]
            Resort_ind = np.max(Resort_ind)
            
            Kpath1_array = Kpath1[ind][:Resort_ind]
            pBNAD1_array = pBNAD1_data[ind][:Resort_ind]
            if Kpath1_array[1]<Kpath1_array[0]:
                Kpath1_array = Kpath1[ind][Resort_ind:]
                Resort_Kpath1 = np.append(Kpath1_array[:-1] + K_shift, Kpath1[ind])
                
                pBNAD1_array = pBNAD1_data[ind][Resort_ind:]
                Resort_pBAND1 = np.r_[pBNAD1_array[:-1], pBNAD1_data[ind]]
            
            else:
                Resort_Kpath1 = np.append(Kpath1[ind], Kpath1_array[1:] + K_shift)
                Resort_pBAND1 = np.r_[pBNAD1_data[ind], pBNAD1_array[1:]]
            # ==========
            Resort_ind = np.where(np.abs(Kpath2[ind]-sepK) < error)[0]
            Resort_ind = np.max(Resort_ind)
            
            Kpath2_array = Kpath2[ind][:Resort_ind]
            pBNAD2_array = pBNAD2_data[ind][:Resort_ind]

            if Kpath2_array[1]<Kpath2_array[0]:
                Kpath2_array = Kpath2[ind][Resort_ind:]
                Resort_Kpath2 = np.append(Kpath2_array[:-1] + K_shift, Kpath2[ind])
                
                pBNAD2_array = pBNAD2_data[ind][Resort_ind:]
                Resort_pBAND2 = np.r_[pBNAD2_array[:-1], pBNAD2_data[ind]]
            
            else:
                Resort_Kpath2 = np.append(Kpath2[ind], Kpath2_array[1:] + K_shift)
                Resort_pBAND2 = np.r_[pBNAD2_data[ind], pBNAD2_array[1:]]

            new_Kpath1 = np.append(new_Kpath1, Resort_Kpath1)
            new_pBAND1 = np.append(new_pBAND1, Resort_pBAND1)
            new_Kpath2 = np.append(new_Kpath2, Resort_Kpath2)
            new_pBAND2 = np.append(new_pBAND2, Resort_pBAND2)
        
        new_Kpath1 = new_Kpath1.reshape(NBAND, len(new_Kpath1)//NBAND)
        new_pBAND1 = new_pBAND1.reshape(NBAND, len(new_pBAND1)//NBAND//2, 2)

        new_Kpath2 = new_Kpath2.reshape(NBAND, len(new_Kpath2)//NBAND)
        new_pBAND2 = new_pBAND2.reshape(NBAND, len(new_pBAND2)//NBAND//2, 2)
        
        BAND1output  = (new_Kpath1, new_pBAND1)
        BAND2output  = (new_Kpath2, new_pBAND2)
        LABELoutput  = (Resort_labels, Resort_ticks)

        (self.labels, self.ticks) = (Resort_labels, Resort_ticks)
        (self.Kpath1, self.pBNAD1_data) = (new_Kpath1, new_pBAND1)
        (self.Kpath2, self.pBNAD2_data) = (new_Kpath2, new_pBAND2)

        return BAND1output, BAND2output, LABELoutput
# ==========
    def Edit_Interpolation(self):
        NBAND = self.NBAND

        labels   = self.labels
        ticks    = self.ticks
        
        Kpath1      = self.Kpath1
        pBNAD1_data = self.pBNAD1_data
        
        Kpath2      = self.Kpath2
        pBNAD2_data = self.pBNAD2_data

        new_Kpath1 = np.array([])
        new_Kpath2 = np.array([])
        new_pBAND1 = np.array([])
        new_pBAND2 = np.array([])
        for ind in range(NBAND):
            
            f_inter_data   = interp1d(Kpath1[ind], pBNAD1_data[ind][:,0])
            f_inter_weight = interp1d(Kpath1[ind], pBNAD1_data[ind][:,1])
            # f_interpolate_2d = interp2d(Kpath1, pBNAD1_data[ind][:,0], pBNAD1_data[ind][:,1], kind='cubic')

            inte_Kpath1 = np.linspace(Kpath1[ind][0], Kpath1[ind][-1], 1000)
            new_Kpath1  = np.append(new_Kpath1, inte_Kpath1)
            
            inte_pBAND1 = np.c_[f_inter_data(inte_Kpath1), f_inter_weight(inte_Kpath1)]
            new_pBAND1  = np.append(new_pBAND1, inte_pBAND1)
            # ==========
            f_inter_data   = interp1d(Kpath2[ind], pBNAD2_data[ind][:,0])
            f_inter_weight = interp1d(Kpath2[ind], pBNAD2_data[ind][:,1])

            inte_Kpath2 = np.linspace(Kpath2[ind][0], Kpath2[ind][-1], 1000)
            new_Kpath2  = np.append(new_Kpath2, inte_Kpath2)
            
            inte_pBAND2 = np.c_[f_inter_data(inte_Kpath2), f_inter_weight(inte_Kpath2)]
            new_pBAND2  = np.append(new_pBAND2, inte_pBAND2)
        
        new_Kpath1 = new_Kpath1.reshape(NBAND, len(new_Kpath1)//NBAND)
        new_pBAND1 = new_pBAND1.reshape(NBAND, len(new_pBAND1)//NBAND//2, 2)

        new_Kpath2 = new_Kpath2.reshape(NBAND, len(new_Kpath2)//NBAND)
        new_pBAND2 = new_pBAND2.reshape(NBAND, len(new_pBAND2)//NBAND//2, 2)
        
        
        BAND1output  = (new_Kpath1, new_pBAND1)
        BAND2output  = (new_Kpath2, new_pBAND2)
        LABELoutput  = (labels, ticks)
        return BAND1output, BAND2output, LABELoutput
# >>>>>>>>>> <<<<<<<<<<
class procarBNADplot():
    def __init__(self, FilePath:str, fermiEnergy=0, PROCARtype='vasp'):
        
        self.parser   = ppr.io.Parser(code=PROCARtype, dir=FilePath)
        self.bandData = self.parser.ebs.bands[:,:,0]-fermiEnergy
        
        self.kpathPos = self.parser.ebs.kpath.tick_positions
        self.kpathLab = self.parser.ebs.kpath.tick_names
        
        self.kpath_CarPos_x = self.parser.ebs.kpoints_cartesian[:,0]
        self.kpath_CarPos_y = self.parser.ebs.kpoints_cartesian[:,1]

        self.nband   = self.parser.ebs.nbands
        self.nkpoint = self.parser.ebs.nkpoints
# ==========    
    def Read_AllData_HighSymmPath(self):
        
        kpathPos = self.kpathPos
        kpath_CarPos_x = self.kpath_CarPos_x
        kpath_CarPos_y = self.kpath_CarPos_y
        
        HighSymmPoint_x = np.array([])
        HighSymmPoint_y = np.array([])
        for kpos in kpathPos:
            HighSymmPoint_x = np.append(HighSymmPoint_x, kpath_CarPos_x[kpos])
            HighSymmPoint_y = np.append(HighSymmPoint_y, kpath_CarPos_y[kpos])
        shift_HighSymmPoint_x = np.delete(HighSymmPoint_x, 0)
        shift_HighSymmPoint_y = np.delete(HighSymmPoint_y, 0)
        
        Dist_HighSymmPoint = np.sqrt((shift_HighSymmPoint_x-HighSymmPoint_x[:-1])**2 + (shift_HighSymmPoint_y-HighSymmPoint_y[:-1])**2)

        Dist_HighSymmPoint = np.insert(Dist_HighSymmPoint, 0, 0)
        HighSymmPoint_ticks = np.cumsum(Dist_HighSymmPoint)
        HighSymmPoint_label = self.kpathLab

        self.HighSymmPoint_ticks = HighSymmPoint_ticks
        self.HighSymmPoint_label = HighSymmPoint_label

        Tools.Process_Word("# >>>>>>>>>> High Symmetry Path <<<<<<<<<< #")
        print(HighSymmPoint_label)
        print(HighSymmPoint_ticks)

        LABELoutput = (HighSymmPoint_label, HighSymmPoint_ticks)
        
        return LABELoutput
# ==========
    def Read_AllData_Band(self):
        
        bandData = self.bandData
        bandData = np.transpose(bandData)

        HighSymmPoint_ticks = self.HighSymmPoint_ticks
        nbandPath = self.nkpoint//(len(HighSymmPoint_ticks)-1)
        
        kpathData = np.array([])
        for i in range(len(HighSymmPoint_ticks)-1):
            kpathData = np.append(kpathData,
                                  np.linspace(HighSymmPoint_ticks[i], HighSymmPoint_ticks[i+1], nbandPath)
                                  )
        
        Tools.Process_Word("# >>>>>>>>>> Band <<<<<<<<<< #")
        print("# ==========")
        print("(kpoints, bands, total)")
        print(np.shape(bandData))
        print("# ==========")
        print(np.shape(kpathData))

        BANDoutput = (kpathData, bandData)

        self.kpathData = kpathData

        kwargs_plot1 = dict(
            color='k',
            linewidth=0.5
        )

        return BANDoutput, kwargs_plot1
# ==========    
    def Read_AllData_projectionData(self):
        projData  = self.parser.ebs.projected[:,:,:,0,:,:]
        self.projData = projData
        
        self.natoms    = self.parser.ebs.natoms
        self.norbitals = self.parser.ebs.norbitals
        self.nspins    = self.parser.ebs.nspins
        
        self.atomData    = np.transpose(projData[:,:,:,:,0].sum(axis=3))
        self.orbitalData = np.transpose(projData.sum(axis=(2, 4)))
        self.spinData    = np.transpose(projData.sum(axis=(2, 3)))

        Tools.Check_out_Word("#>>>>> Read project band data <<<<<#")
        Tools.Process_Word("# =====")
        print(f"There are {self.nband} bands of each kpoint")
        print(f"There are {self.nkpoint} kpoints of each band")
        Tools.Process_Word("# =====")
        print(f"The shape of the projection data: {np.shape(projData)}")
        print(f"There are {self.natoms} atoms in this data")
        print(f"There are {self.norbitals} orbitals in this data")
        print(f"There are {self.nspins} spins in this data")
        print(f"[0 => total spin density; 1 => Sx; 2 => Sy; 3 => Sz]")

        PROJoutput = (self.atomData, self.orbitalData, self.spinData)

        return PROJoutput
# ==========    
    def Read_SpinData_projectionData(self):
        projData  = self.parser.ebs.projected[:,:,:,0,:,:]
        spinData  = np.transpose(projData.sum(axis=(2, 3)))
        print(np.shape(spinData))
        kwargs_spin = dict(
            s=50,
            marker='.',
            cmap='seismic',
            norm=colors.Normalize(-0.5, 0.5),
            alpha=1.0,
            edgecolor='none',
        )
        return spinData, kwargs_spin
# ==========    
    def Read_OrbitalData_projectionData(self, ):
        projData  = self.parser.ebs.projected[:,:,:,0,:,:]
        orbitalData = np.transpose(projData.sum(axis=(2, 4)))
        kwargs_orbital = dict(
            s=50,
            marker='.',
            cmap='Blues',
            norm=colors.Normalize(0, 1),
            alpha=1.0,
            edgecolor='none',
        )
        return orbitalData, kwargs_orbital
# ==========    
    def Read_AtomData_projectionData(self, AtomList:list):
        projData = self.parser.ebs.projected[:,:,:,0,:,:]
        atomData = projData[:,:,:,:,0].sum(axis=3)[:,:,AtomList].sum(axis=2, keepdims=True)
        atomData = np.transpose(atomData)
        kwargs_orbital = dict(
            s=50,
            marker='.',
            cmap='Blues',
            norm=colors.Normalize(0, 1),
            alpha=1.0,
            edgecolor='none',
        )
        return atomData, kwargs_orbital
# ==========    
    def Read_AtomCompData_projectionData(self, AtomList1:list, AtomList2:list, type="1-2"):
        projData = self.parser.ebs.projected[:,:,:,0,:,:]
        
        atomData1 = projData[:,:,:,:,0].sum(axis=3)[:,:,AtomList1].sum(axis=2, keepdims=True)
        atomData2 = projData[:,:,:,:,0].sum(axis=3)[:,:,AtomList2].sum(axis=2, keepdims=True)
        
        match type:
            case "1-2":
                atomData = np.transpose(atomData1-atomData2)
            case "2-1":
                atomData = np.transpose(atomData2-atomData1)
            case _:
                Tools.Check_out_Word("No this kind of type")

        kwargs_atomcomp = dict(
            s=50,
            marker='.',
            cmap='jet',
            norm=colors.Normalize(-1, 1),
            alpha=1.0,
            edgecolor='none',
        )
              
        return atomData, kwargs_atomcomp
# ==========
    def Plot_projectTools(self, PlotData, kwargs_plot1:dict, kwargs_plot2:dict, Setting:tuple):
        K_path, B_data = self.kpathData, np.transpose(self.bandData)
        L_ticks = self.HighSymmPoint_ticks
        L_label = self.HighSymmPoint_label

        plt.figure(figsize=(5, 4))
        plt.title(Setting[0])
        for i in range(108):
            plt.plot(K_path, B_data[i],
                    **kwargs_plot1
                    )
            plt.scatter(
                    K_path, B_data[i],
                    c=PlotData[Setting[1]][i],
                    **kwargs_plot2
                )
            
        plt.ylim(-1, 1)
        plt.xlim(np.min(K_path), np.max(K_path))
        plt.vlines(x=L_ticks, ymin=-1, ymax=1, colors=color[-1])
        plt.hlines(y=0, xmin=np.min(K_path)-0.01, xmax=np.max(K_path)+0.01, colors=color[-1])
        plt.xticks(ticks=L_ticks, labels=L_label)
        plt.colorbar()
        plt.show()
# >>>>>>>>>> <<<<<<<<<<
if __name__ == '__main__':

    kwargs_plot = dict(
                    s=100,
                    marker='.',
                    cmap='seismic',
                    norm=colors.Normalize(-1, 1),
                    alpha=0.5,
                    edgecolor='none',
                )
        
    def Normal_vaspkitBandtest():
        examp = Normal_vaspkitBANDplot(B_FilePath=r"testingData\nBAND\Pd1Te2_socBAND.dat",
                                    L_FilePath=r"testingData\nBAND\Pd1Te2_socLABEL.dat",
                                    )
        # =====
        result  = examp.Read_AllData()
        (K_path, B_data), (L_label, L_ticks) = result
        # =====
        result2 = examp.Read_SpecKData(startK=L_ticks[0], endK=L_ticks[3])
        (K_path, B_data), (L_label, L_ticks) = result2
        # =====
        result3 = examp.Edit_ResortData(BANDinput=(K_path, B_data), LABELinput=(L_label, L_ticks), sepK=L_ticks[2])
        (K_path, B_data), (L_label, L_ticks) = result3
        
        plt.figure(figsize=(5, 4))
        for data in B_data:
            plt.plot(K_path, data, c=color[0], linewidth=3)
        plt.xticks(L_ticks, L_label, fontsize=15)
        plt.xlim(np.min(K_path), np.max(K_path))
        # =====
        plt.ylabel("Energy (eV)", fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-3, 3)
        # ===========================================================
        plt.hlines(y=0, xmin=np.min(K_path), xmax=np.max(K_path), 
                colors=color[-1],
                linestyles='--')
        plt.vlines(x=L_ticks, ymin=-10, ymax=10, 
                colors=color[-1],
                linestyles='--')
        
        plt.show()
    
    def Project_vaspkitBandtest():
        kwargs_plot2 = dict(
            s=50,
            marker='.',
            cmap='Blues',
            norm=colors.Normalize(0, 1),
            alpha=1.0,
            edgecolor='none',
        )
        band_num = 224
        examp = Project_vaspkitBANDplot(NBAND = band_num,
                                    pB1_FilePath = r"testingData\patomBAND\2_BAND_dn\PBAND_SUM_SOC_SOC.dat",
                                    pB2_FilePath = r"testingData\patomBAND\3_BAND_up\PBAND_SUM_SOC_SOC.dat",
                                     pL_FilePath = r"testingData\patomBAND\2_BAND_dn\KLABELS",
                                    )
        result = examp.Read_AllData(p=False)
        (K1_path, pB1_data), (K2_path, pB2_data), (L_label, L_ticks) = result
        # =====
        # result2 = examp.Edit_Interpolation()
        # (K1_path, pB1_data), (K2_path, pB2_data), (L_label, L_ticks) = result2
        # =====
        result3 = examp.Read_SpecKData(startK=L_ticks[0], endK=L_ticks[3])
        (K1_path, pB1_data), (K2_path, pB2_data), (L_label, L_ticks) = result3
        # =====
        result4 = examp.Edit_ResortData(BANDinput1=(K1_path, pB1_data), BANDinput2=(K2_path, pB2_data), LABELinput=(L_label, L_ticks), sepK=L_ticks[2])
        (K1_path, pB1_data), (K2_path, pB2_data), (L_label, L_ticks) = result4
        # =====
        print(np.shape(pB1_data))
        plt.figure(figsize=(4, 3))
        
        for ind in range(band_num):
            plt.plot(K1_path[ind], pB1_data[ind][:,0], 
                    color = 'k', linewidth = 0.5)
            plt.scatter(K1_path[ind], pB1_data[ind][:,0], 
                    c=pB1_data[ind][:,1]*(1), linewidth=3,
                    **kwargs_plot2)
            # -----
            # plt.plot(K2_path[ind], pB2_data[ind][:,0], 
            #         color = 'k', linewidth = 0.5)
            # plt.scatter(K2_path[ind], pB2_data[ind][:,0], 
            #         c=pB1_data[ind][:,1], linewidth=3,
            #         **kwargs_plot2)

        plt.colorbar()    
        plt.xticks(L_ticks, L_label, fontsize=15)
        plt.xlim(np.min(K1_path), np.max(K1_path))
        # =====
        plt.ylabel("Energy (eV)", fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-3, 3)
        # ===========================================================
        plt.hlines(y=0, xmin=np.min(K1_path), xmax=np.max(K1_path), 
                colors=color[-1],
                linestyles='--')
        plt.vlines(x=L_ticks, ymin=-10, ymax=10, 
                colors=color[-1],
                linestyles='--')
        
        # plt.savefig("test2.png")
        plt.tight_layout()
        plt.show()

    def Normal_procarBandtest():
        examp = procarBNADplot(FilePath='testingData/vaspPROCARtest', PROCARtype='vasp')
        (L_label, L_ticks) = examp.Read_AllData_HighSymmPath()
        (K_path, B_data)   = examp.Read_AllData_Band()

        plt.figure(figsize=(5, 4))
        for i in range(108):
            plt.plot(K_path, B_data[:,i][:,0]-0.6411, c=color[0])
        plt.ylim(-1, 1)
        plt.xlim(np.min(K_path), np.max(K_path))
        plt.vlines(x=L_ticks, ymin=-1, ymax=1, colors=color[-1])
        plt.xticks(ticks=L_ticks, labels=L_label)
        # plt.savefig("Test_Cr3Te4-PtTe2_pyprocar.png",
        #             transparent=True, bbox_inches='tight')
        plt.show()


# >>>>>>>>>> Runing <<<<<<<<<< #
    Project_vaspkitBandtest()
    # =====
    # Normal_procarBandtest()