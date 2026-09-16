import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pyprocar as ppr
from STT_Tool import Process_Word as pw
from STT_Tool import Tools
from scipy.interpolate import interp1d

color = Tools.ColorList()

# >>>>>>>>>> Info Block <<<<<<<<<< #
def _Info_():
    print("=====")
    print("We have the following methods:")
    methodList = ("Read_AllData_Band", 
                  "Find_BandGap",
                  "Edit_interpolation_1DBand",
                  "Read_AllData_projectionData", 
                  "kwargsList",
                  "Read_SpinData_projectionData",
                  "Read_OrbitalData_projectionData",
                  "Read_AtomData_projectionData",
                  "Read_AtomCompData_projectionData",
                  "Read_OrbitalCompData_projectionData",
                  "Read_Custom_projectionData",
                  "Plot_projectTools",
                  )
    for name in methodList:
        pw(f"    {name}") 
    print("=====") 
# >>>>>>>>>> <<<<<<<<<< #

class procarBNADplot():
    def __init__(self, FilePath:str, fermiEnergy=0, PROCARtype='vasp'):

        print(">>>>> pyprocar verision <<<<<")
        print(ppr.__version__)
        print(">>>>> =============== <<<<<")
        print()
        
        self.parser   = ppr.io.Parser(code=PROCARtype, dirpath=FilePath)
        self.bandData = self.parser.ebs.bands[:,:,0]-fermiEnergy
        self.kpoints  = self.parser.ebs.kpoints_cartesian

        self.kpathPos = self.parser.ebs.kpath.tick_positions
        self.kpathLab = self.parser.ebs.kpath.tick_names

        self.nband   = self.parser.ebs.nbands
        self.nkpoint = self.parser.ebs.nkpoints
        self.table = """
+-------+-----+------+------+------+------+------+------+------+------+
|n-lm   |  0  |   1  |  2   |   3  |   4  |   5  |   6  |   7  |   8  |
+=======+=====+======+======+======+======+======+======+======+======+
|-1(tot)|  s  |  py  |  pz  |  px  | dxy  | dyz  | dz2  | dxz  |x2-y2 |
+-------+-----+------+------+------+------+------+------+------+------+
"""
        self.kwargs_line = dict(color='k', linewidth=0.5)
        self.kwargs_seismic = dict(s=50, marker='.', cmap='seismic', norm=colors.Normalize(-0.5, 0.5), alpha=1.0, edgecolor='none')
        self.kwargs_blue = dict(s=50, marker='.', cmap='Blues', norm=colors.Normalize(0, 1), alpha=1.0, edgecolor='none')
        self.kwargs_jet = dict(s=50, marker='.', cmap='jet', norm=colors.Normalize(-1, 1), alpha=1.0, edgecolor='none')

# ==========
    def Read_AllData_Band(self):
        
        bandData = self.bandData
        bandData = np.transpose(bandData)
        # =====
        kpoints = self.kpoints
        k_distance = np.linalg.norm(kpoints[1:,:]-kpoints[:-1,:], axis=1)
        k_distance = np.insert(k_distance, 0, 0)
        GapPos = np.where(k_distance>0.05)[0]

        k_distance = np.delete(k_distance, GapPos)
        k_distance = np.insert(k_distance, GapPos, 0)
        kpathData = np.cumsum(k_distance)
        # =====
        HighSymmPoint_label = self.kpathLab
        self.HighSymmPoint_label = HighSymmPoint_label
        # =====
        HighSymmPoint_ticks = kpathData[self.kpathPos]
        self.HighSymmPoint_ticks = HighSymmPoint_ticks

        Tools.Process_Word("# >>>>>>>>>> Band <<<<<<<<<< #")
        print("# ==========")
        print("(bands, kpoints)")
        print(np.shape(bandData))

        BANDoutput  = (kpathData, bandData)
        LABELoutput = (HighSymmPoint_label, HighSymmPoint_ticks)

        self.kpathData = kpathData

        # Call parameter dynamically
        kwargs_plot1 = self.kwargs_line

        return BANDoutput, LABELoutput, kwargs_plot1

# ==========    
    def Find_BandGap(self):
        B_data = np.transpose(self.bandData)
        Cond_min = np.where(np.all(B_data>0, axis=1))
        Cond_min = Cond_min[0]
        Cond_min = np.min(Cond_min)
        ConductionBand_min = np.min(B_data[Cond_min])
        print("Conduction band minimum is:" , ConductionBand_min)

        Val_max = np.where(np.all(B_data<0, axis=1))
        Val_max = Val_max[0]
        Val_max = np.max(Val_max)
        ValenceBand_max = np.max(B_data[Val_max])
        print("Valence band maximum is:" , ValenceBand_max)

        BandGap = ConductionBand_min-ValenceBand_max
        pw(str(BandGap))

        return ValenceBand_max, ConductionBand_min, BandGap

# ==========
## By Gemini
    def Edit_interpolation_1DBand(self, multiplier=1):
        """
        multiplier: 放大倍率，決定資料密集度要增加幾倍
        """
        self.interp_multiplier = multiplier
        
        # 確保基準的 kpathData 已經被建立
        if not hasattr(self, 'kpathData'):
            self.Read_AllData_Band()
            
        Tools.Check_out_Word("#>>>>> create 1D interpolation <<<<<#")
        Tools.Process_Word("# =====")
        print(f"Interpolating band data with multiplier: {multiplier}")
        
        kpath = self.kpathData
        bandData = np.transpose(self.bandData) # shape: (bands, kpoints)
        
        # 尋找高對稱點造成的斷點 (X軸重複處)，以此為界切分資料
        duplicate_indices = np.where(np.diff(kpath) == 0)[0] + 1
        
        x_segments = np.split(kpath, duplicate_indices)
        y_segments = np.split(bandData, duplicate_indices, axis=1)
        
        new_x_list, new_y_list = [], []
        
        for x_seg, y_seg in zip(x_segments, y_segments):
            if len(x_seg) < 2:
                new_x_list.append(x_seg)
                new_y_list.append(y_seg)
                continue
            
            # 針對每一小段生成更密集的 X 座標
            new_x = np.linspace(x_seg[0], x_seg[-1], len(x_seg) * multiplier)
            
            # 使用三次樣條內插 (cubic)
            f = interp1d(x_seg, y_seg, kind='cubic', axis=1)
            new_y = f(new_x)
            
            new_x_list.append(new_x)
            new_y_list.append(new_y)
            
        # 重新將切段的資料拼接起來
        self.new_kpathData = np.concatenate(new_x_list)
        self.new_bandData = np.concatenate(new_y_list, axis=1)
        
        BANDoutput_1D_Original = (kpath, bandData)
        BANDoutput_1D = (self.new_kpathData, self.new_bandData)
        
        return BANDoutput_1D_Original, BANDoutput_1D

# ==========
## By Gemini
    def _interpolate_1D_projectData(self, projData_1D):
        # 如果使用者沒有呼叫過內插功能，就直接回傳原始資料
        if not hasattr(self, 'new_kpathData'):
            return projData_1D
            
        kpath = self.kpathData
        duplicate_indices = np.where(np.diff(kpath) == 0)[0] + 1
        
        x_segments = np.split(kpath, duplicate_indices)
        
        # ---> 【修改這裡】將 axis=1 改為 axis=-1，確保永遠切分 kpoints 維度 <---
        y_segments = np.split(projData_1D, duplicate_indices, axis=-1)
        
        new_y_list = []
        for x_seg, y_seg in zip(x_segments, y_segments):
            if len(x_seg) < 2:
                new_y_list.append(y_seg)
                continue
                
            new_x = np.linspace(x_seg[0], x_seg[-1], len(x_seg) * self.interp_multiplier)
            
            # ---> 【修改這裡】將 axis=1 改為 axis=-1 <---
            f = interp1d(x_seg, y_seg, kind='cubic', axis=-1)
            new_y = f(new_x)
            
            new_y_list.append(new_y)
            
        # ---> 【修改這裡】將 axis=1 改為 axis=-1 <---
        return np.concatenate(new_y_list, axis=-1)

# ==========    
    def Read_AllData_projectionData(self):
        projData  = self.parser.ebs.projected[:,:,:,0,:,:]
        self.projData = projData
        
        self.natoms    = self.parser.ebs.natoms
        self.norbitals = self.parser.ebs.norbitals
        self.nspins    = self.parser.ebs.nspins
        
        Tools.Check_out_Word("#>>>>> Read project band data <<<<<#")
        Tools.Process_Word("# =====")
        print(f"There are {self.nband} bands of each kpoint")
        print(f"There are {self.nkpoint} kpoints of each band")
        Tools.Process_Word("# =====")
        print(f"The shape of the projection data: {np.shape(projData)}")
        print(f"There are {self.natoms} atoms in this data")
        print(f"There are {self.norbitals} orbitals in this data")
        print(f"There are {self.nspins} spins in this data")
        if self.nspins == 4:
            print(f"[0 => total spin density; 1 => Sx; 2 => Sy; 3 => Sz]")
        elif self.nspins == 2:
            print("SpinsList is useless, all projection results will be split into")
            print("[0 => spin up; 1 => spin down]")

# ========== 
    def kwargsList(self):
        return self.kwargs_line, self.kwargs_seismic, self.kwargs_blue, self.kwargs_jet

# ==========    
    def Read_SpinData_projectionData(self, spinList=(0,)):
        spinData = self.parser.ebs.ebs_sum(atoms=None, orbitals=None, spins=spinList)
        spinData = np.transpose(spinData)
        
        spinData = self._interpolate_1D_projectData(spinData)
        
        kwargs_spin = self.kwargs_seismic
        return spinData, kwargs_spin

# ==========    
    def Read_OrbitalData_projectionData(self, orbitalList=None, table=0):
        orbitalData = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitalList, spins=(0,))
        orbitalData = np.transpose(orbitalData)

        orbitalData = self._interpolate_1D_projectData(orbitalData)

        kwargs_orbital = self.kwargs_blue
        if table:
            print(self.table)
        return orbitalData, kwargs_orbital

# ==========    
    def Read_AtomData_projectionData(self, atomList=None):
        atomData = self.parser.ebs.ebs_sum(atoms=atomList, orbitals=None, spins=(0,))
        atomData = np.transpose(atomData)

        atomData = self._interpolate_1D_projectData(atomData)

        kwargs_atom = self.kwargs_blue
        return atomData, kwargs_atom

# ==========    
    def Read_AtomCompData_projectionData(self, atomList1:list, atomList2:list, type="1-2"):
        atomData1 = self.parser.ebs.ebs_sum(atoms=atomList1, orbitals=None, spins=(0,))
        atomData2 = self.parser.ebs.ebs_sum(atoms=atomList2, orbitals=None, spins=(0,))
        
        if type == "1-2":
            atomData = atomData1 - atomData2
        elif type == "2-1":
            atomData = atomData2 - atomData1
        else:
            Tools.Check_out_Word("No this kind of type")

        atomData = np.transpose(atomData)
        atomData = self._interpolate_1D_projectData(atomData)

        # Call parameter dynamically
        kwargs_atomcomp = self.kwargs_jet
        return atomData, kwargs_atomcomp

# ==========  
    def Read_OrbitalCompData_projectionData(self, orbitList1:list, orbitList2:list, type="1-2"):
        orbitalData1 = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitList1, spins=(0,))
        orbitalData2 = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitList2, spins=(0,))
        
        if type == "1-2":
            orbitalData = orbitalData1 - orbitalData2
        elif type == "2-1":
            orbitalData = orbitalData2 - orbitalData1
        else:
            Tools.Check_out_Word("No this kind of type")

        orbitalData = np.transpose(orbitalData)
        orbitalData = self._interpolate_1D_projectData(orbitalData)

        # Call parameter dynamically
        kwargs_otbitcomp = self.kwargs_jet
        return orbitalData, kwargs_otbitcomp

# ==========
    def Read_Custom_projectionData(self, atomList=None, orbitalList=None, spinList=(0,)):
        projData = self.parser.ebs.ebs_sum(atoms=atomList, orbitals=orbitalList, spins=spinList)
        projData = np.transpose(projData)
        
        projData = self._interpolate_1D_projectData(projData)

        # Call parameter dynamically
        kwargs_spin = self.kwargs_seismic
        kwargs_other = self.kwargs_blue

        return projData, (kwargs_spin, kwargs_other)

# ==========
    def Plot_projectTools(self, PlotData, kwargs_plot1:dict, kwargs_plot2:dict, Title:str, 
                          yboundary:tuple, 
                          xboundary=None
                          ):
        
        if hasattr(self, 'new_kpathData'):
            K_path, B_data = self.new_kpathData, self.new_bandData
        else:
            K_path, B_data = self.kpathData, np.transpose(self.bandData)
            
        L_ticks = self.HighSymmPoint_ticks
        L_label = self.HighSymmPoint_label
        
        plt.figure(figsize=(5, 4))
        plt.title(Title)
        for i in range(self.nband):
            dB = np.abs(B_data[i]-B_data[i-1])
            dBBool = np.all(dB<1e-2)
            if dBBool:
                proj = PlotData[i] + PlotData[i-1]
            else:
                proj =  PlotData[i]
            plt.plot(K_path, B_data[i], **kwargs_plot1)
            plt.scatter(K_path, B_data[i], c=proj, **kwargs_plot2)
            
        plt.vlines(x=L_ticks, ymin=yboundary[0]-1, ymax=yboundary[1]+1, colors=color[-1])
        plt.hlines(y=0, xmin=np.min(K_path)-0.01, xmax=np.max(K_path)+0.01, colors=color[-1])
        plt.xticks(ticks=L_ticks, labels=L_label)
        plt.ylim(yboundary[0], yboundary[1])
        if xboundary==None:
            plt.xlim(np.min(K_path), np.max(K_path))
        else:
            plt.xlim(xboundary[0], xboundary[1])
        plt.colorbar()
        plt.show()
# >>>>>>>>>> <<<<<<<<<<