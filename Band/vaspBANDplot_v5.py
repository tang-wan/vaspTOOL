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
    def __init__(self, FilePath:str, fermiEnergy=0, PROCARtype='vasp', spin_mode='all'):
        """
        spin_mode: 'all', 'up', or 'down'. Determines which spins to process/plot for collinear calculations.
        """
        print(">>>>> pyprocar verision <<<<<")
        print(ppr.__version__)
        print(">>>>> =============== <<<<<")
        print()
        
        self.parser   = ppr.io.Parser(code=PROCARtype, dirpath=FilePath)
        print(np.shape(self.parser.ebs.bands))
        self.spin_mode = spin_mode.lower()
                
        # Get the shape of raw band data (nkpoints, nbands, nspins)
        raw_bands = self.parser.ebs.bands
        n_spins = raw_bands.shape[2]
        
        if n_spins == 1:
            # Handle SOC or non-spin-polarized (Non-collinear or Non-spin-polarized)
            self.bandData = raw_bands[:, :, 0] - fermiEnergy
            self.nband = self.parser.ebs.nbands
        elif n_spins == 2:
            # Handle Collinear (CL) spin-polarized:
            # raw_bands[:, :, 0] is Spin Up
            # raw_bands[:, :, 1] is Spin Down
            spin_up = raw_bands[:, :, 0] - fermiEnergy
            spin_down = raw_bands[:, :, 1] - fermiEnergy
            
            if self.spin_mode == 'up':
                self.bandData = spin_up
                self.nband = self.parser.ebs.nbands
            elif self.spin_mode == 'dn':
                self.bandData = spin_down
                self.nband = self.parser.ebs.nbands
            else:
                # We concatenate Spin Down bands after Spin Up, resulting in shape (nkpoints, nbands * 2)
                # Concatenate along axis=1 (nbands)
                self.bandData = np.concatenate((spin_up, spin_down), axis=1)
                # Since both spins are concatenated, total number of bands is doubled
                self.nband = self.parser.ebs.nbands * 2
        else:
            raise ValueError(f"Unexpected spin dimension: {n_spins}")

        self.kpoints  = self.parser.ebs.kpoints_cartesian

        self.kpathPos = self.parser.ebs.kpath.tick_positions
        self.kpathLab = self.parser.ebs.kpath.tick_names

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
        multiplier: Magnification factor, determines how much the data density increases
        """
        self.interp_multiplier = multiplier
        
        # Ensure the baseline kpathData has been established
        if not hasattr(self, 'kpathData'):
            self.Read_AllData_Band()
            
        Tools.Check_out_Word("#>>>>> create 1D interpolation <<<<<#")
        Tools.Process_Word("# =====")
        print(f"Interpolating band data with multiplier: {multiplier}")
        
        kpath = self.kpathData
        bandData = np.transpose(self.bandData) # shape: (bands, kpoints)
        
        # Find breakpoints caused by high-symmetry points (duplicate X-axis values) to split data
        duplicate_indices = np.where(np.diff(kpath) == 0)[0] + 1
        
        x_segments = np.split(kpath, duplicate_indices)
        y_segments = np.split(bandData, duplicate_indices, axis=1)
        
        new_x_list, new_y_list = [], []
        
        for x_seg, y_seg in zip(x_segments, y_segments):
            if len(x_seg) < 2:
                new_x_list.append(x_seg)
                new_y_list.append(y_seg)
                continue
            
            # Generate denser X coordinates for each segment
            new_x = np.linspace(x_seg[0], x_seg[-1], len(x_seg) * multiplier)
            
            # Use cubic spline interpolation
            f = interp1d(x_seg, y_seg, kind='cubic', axis=1)
            new_y = f(new_x)
            
            new_x_list.append(new_x)
            new_y_list.append(new_y)
            
        # Re-concatenate the segmented data
        self.new_kpathData = np.concatenate(new_x_list)
        self.new_bandData = np.concatenate(new_y_list, axis=1)
        
        BANDoutput_1D_Original = (kpath, bandData)
        BANDoutput_1D = (self.new_kpathData, self.new_bandData)
        
        return BANDoutput_1D_Original, BANDoutput_1D

# ==========
## By Gemini
    def _interpolate_1D_projectData(self, projData_1D):
        # Directly return original data if the user has not called the interpolation function
        if not hasattr(self, 'new_kpathData'):
            return projData_1D
            
        kpath = self.kpathData
        duplicate_indices = np.where(np.diff(kpath) == 0)[0] + 1
        
        x_segments = np.split(kpath, duplicate_indices)
        
        # ---> [Modified here] Change axis=1 to axis=-1 to ensure kpoints dimension is always split <---
        y_segments = np.split(projData_1D, duplicate_indices, axis=-1)
        
        new_y_list = []
        for x_seg, y_seg in zip(x_segments, y_segments):
            if len(x_seg) < 2:
                new_y_list.append(y_seg)
                continue
                
            new_x = np.linspace(x_seg[0], x_seg[-1], len(x_seg) * self.interp_multiplier)
            
            # ---> [Modified here] Change axis=1 to axis=-1 <---
            f = interp1d(x_seg, y_seg, kind='cubic', axis=-1)
            new_y = f(new_x)
            
            new_y_list.append(new_y)
            
        # ---> [Modified here] Change axis=1 to axis=-1 <---
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
        n_spins = self.parser.ebs.bands.shape[2]
                
        if n_spins == 1:
            spinData = self.parser.ebs.ebs_sum(atoms=None, orbitals=None, spins=spinList)
        elif n_spins == 2:
            spinData_up = self.parser.ebs.ebs_sum(atoms=None, orbitals=None, spins=None)[:,:,0]
            spinData_down = self.parser.ebs.ebs_sum(atoms=None, orbitals=None, spins=None)[:,:,1]*(-1)
            
            if self.spin_mode == 'up':
                spinData = spinData_up
            elif self.spin_mode == 'down':
                spinData = spinData_down
            else:
                spinData = np.concatenate((spinData_up, spinData_down), axis=1)

        spinData = np.transpose(spinData)
        spinData = self._interpolate_1D_projectData(spinData)
        
        kwargs_spin = self.kwargs_seismic
        return spinData, kwargs_spin

# ==========    
    def Read_OrbitalData_projectionData(self, orbitalList=None, table=0):

        n_spins = self.parser.ebs.bands.shape[2]
                
        if n_spins == 1:
            orbitalData = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitalList, spins=(0,))
        elif n_spins == 2:
            orbitalData_up = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitalList, spins=None)[:,:,0]
            orbitalData_down = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitalList, spins=None)[:,:,1]
            
            if self.spin_mode == 'up':
                orbitalData = orbitalData_up
            elif self.spin_mode == 'down':
                orbitalData = orbitalData_down
            else:
                orbitalData = np.concatenate((orbitalData_up, orbitalData_down), axis=1)

        orbitalData = np.transpose(orbitalData)
        orbitalData = self._interpolate_1D_projectData(orbitalData)

        kwargs_orbital = self.kwargs_blue
        if table:
            print(self.table)
        return orbitalData, kwargs_orbital

# ==========    
    def Read_AtomData_projectionData(self, atomList=None):
        n_spins = self.parser.ebs.bands.shape[2]
                
        if n_spins == 1:
            atomData = self.parser.ebs.ebs_sum(atoms=atomList, orbitals=None, spins=(0,))
        elif n_spins == 2:
            atomData_up = self.parser.ebs.ebs_sum(atoms=atomList, orbitals=None, spins=None)[:,:,0]
            atomData_down = self.parser.ebs.ebs_sum(atoms=atomList, orbitals=None, spins=None)[:,:,1]
            
            if self.spin_mode == 'up':
                atomData = atomData_up
            elif self.spin_mode == 'down':
                atomData = atomData_down
            else:
                atomData = np.concatenate((atomData_up, atomData_down), axis=1)

        atomData = np.transpose(atomData)
        atomData = self._interpolate_1D_projectData(atomData)

        kwargs_atom = self.kwargs_blue
        return atomData, kwargs_atom

# ==========    
    def Read_AtomCompData_projectionData(self, atomList1:list, atomList2:list, type="1-2"):
        n_spins = self.parser.ebs.bands.shape[2]
                
        # Read data of Atom1 and Atom2 respectively, and handle dimensionality logic
        if n_spins == 1:
            atomData1 = self.parser.ebs.ebs_sum(atoms=atomList1, orbitals=None, spins=(0,))
            atomData2 = self.parser.ebs.ebs_sum(atoms=atomList2, orbitals=None, spins=(0,))
        elif n_spins == 2:
            atomData1_up = self.parser.ebs.ebs_sum(atoms=atomList1, orbitals=None, spins=None)[:,:,0]
            atomData1_down = self.parser.ebs.ebs_sum(atoms=atomList1, orbitals=None, spins=None)[:,:,1]
            atomData2_up = self.parser.ebs.ebs_sum(atoms=atomList2, orbitals=None, spins=None)[:,:,0]
            atomData2_down = self.parser.ebs.ebs_sum(atoms=atomList2, orbitals=None, spins=None)[:,:,1]
            
            if self.spin_mode == 'up':
                atomData1 = atomData1_up
                atomData2 = atomData2_up
            elif self.spin_mode == 'down':
                atomData1 = atomData1_down
                atomData2 = atomData2_down
            else:
                atomData1 = np.concatenate((atomData1_up, atomData1_down), axis=1)
                atomData2 = np.concatenate((atomData2_up, atomData2_down), axis=1)
        
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

        n_spins = self.parser.ebs.bands.shape[2]
                        
        # Read data of Orbit1 and Orbit2 respectively, and handle dimensionality logic
        if n_spins == 1:
            orbitalData1 = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitList1, spins=(0,))
            orbitalData2 = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitList2, spins=(0,))
        elif n_spins == 2:
            orbitalData1_up = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitList1, spins=None)[:,:,0]
            orbitalData1_down = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitList1, spins=None)[:,:,1]
            orbitalData2_up = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitList2, spins=None)[:,:,0]
            orbitalData2_down = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitList2, spins=None)[:,:,1]
            
            if self.spin_mode == 'up':
                orbitalData1 = orbitalData1_up
                orbitalData2 = orbitalData2_up
            elif self.spin_mode == 'down':
                orbitalData1 = orbitalData1_down
                orbitalData2 = orbitalData2_down
            else:
                orbitalData1 = np.concatenate((orbitalData1_up, orbitalData1_down), axis=1)
                orbitalData2 = np.concatenate((orbitalData2_up, orbitalData2_down), axis=1)
        
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
        n_spins = self.parser.ebs.bands.shape[2]
                        
        if n_spins == 1:
            projData = self.parser.ebs.ebs_sum(atoms=atomList, orbitals=orbitalList, spins=spinList)
        elif n_spins == 2:
            
            projData_up = self.parser.ebs.ebs_sum(atoms=atomList, orbitals=orbitalList, spins=None)[:,:,0]
            projData_down = self.parser.ebs.ebs_sum(atoms=atomList, orbitals=orbitalList, spins=None)[:,:,1]
            
            if self.spin_mode == 'up':
                projData = projData_up
            elif self.spin_mode == 'down':
                projData = projData_down
            else:
                projData = np.concatenate((projData_up, projData_down), axis=1)

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