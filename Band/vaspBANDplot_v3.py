import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pyprocar as ppr

from STT_Tool import Tools
color = Tools.ColorList()
# >>>>>>>>>> <<<<<<<<<<
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

        kwargs_plot1 = dict(
            color='k',
            linewidth=0.5
        )

        return BANDoutput, LABELoutput, kwargs_plot1
# ==========    
    def Read_AllData_projectionData(self):
        projData  = self.parser.ebs.projected[:,:,:,0,:,:]
        self.projData = projData
        
        self.natoms    = self.parser.ebs.natoms
        self.norbitals = self.parser.ebs.norbitals
        self.nspins    = self.parser.ebs.nspins
        
        # self.atomData    = np.transpose(projData[:,:,:,:,0].sum(axis=3))
        # self.orbitalData = np.transpose(projData.sum(axis=(2, 4)))
        # self.spinData    = np.transpose(projData.sum(axis=(2, 3)))

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
            print(f"[0 => spin up; 1 => spin down]")

        # PROJoutput = (self.atomData, self.orbitalData, self.spinData)

        # return PROJoutput
# ==========    
    def Read_SpinData_projectionData(self, spinList=(0,)):
        spinData = self.parser.ebs.ebs_sum(atoms=None, 
                                        orbitals=None, 
                                           spins=spinList
                                           )
        spinData = np.transpose(spinData)
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
    def Read_OrbitalData_projectionData(self, orbitalList=None, table=0):
        orbitalData = self.parser.ebs.ebs_sum(atoms=None, 
                                           orbitals=orbitalList, 
                                              spins=(0,)
                                              )
        orbitalData = np.transpose(orbitalData)
        kwargs_orbital = dict(
            s=50,
            marker='.',
            cmap='Blues',
            norm=colors.Normalize(0, 1),
            alpha=1.0,
            edgecolor='none',
        )
        if table:
            print(self.table)
        return orbitalData, kwargs_orbital
# ==========    
    def Read_AtomData_projectionData(self, atomList=None):
        atomData = self.parser.ebs.ebs_sum(atoms=atomList, 
                                        orbitals=None, 
                                           spins=(0,)
                                           )
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
    def Read_AtomCompData_projectionData(self, atomList1:list, atomList2:list, type="1-2"):
        atomData1 = self.parser.ebs.ebs_sum(atoms=atomList1, orbitals=None, spins=(0,))
        atomData2 = self.parser.ebs.ebs_sum(atoms=atomList2, orbitals=None, spins=(0,))
        
        match type:
            case "1-2":
                atomData = atomData1-atomData2
            case "2-1":
                atomData = atomData2-atomData1
            case _:
                Tools.Check_out_Word("No this kind of type")
        atomData = np.transpose(atomData)

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
    def Read_Custom_projectionData(self, atomList=None, orbitalList=None, spinList=(0,)):
        projData = self.parser.ebs.ebs_sum(atoms=atomList, 
                                        orbitals=orbitalList, 
                                           spins=spinList
                                           )
        projData = np.transpose(projData)
        kwargs_spin = dict(
            s=50,
            marker='.',
            cmap='seismic',
            norm=colors.Normalize(-0.5, 0.5),
            alpha=1.0,
            edgecolor='none',
        )
        kwargs_other = dict(
            s=50,
            marker='.',
            cmap='Blues',
            norm=colors.Normalize(0, 1),
            alpha=1.0,
            edgecolor='none',
        )
        return projData, (kwargs_spin, kwargs_other)
# ==========
    def Plot_projectTools(self, PlotData, kwargs_plot1:dict, kwargs_plot2:dict, Title:str, boundary:tuple):
        K_path, B_data = self.kpathData, np.transpose(self.bandData)
        PlotData = PlotData[0]

        L_ticks = self.HighSymmPoint_ticks
        L_label = self.HighSymmPoint_label
        
        plt.figure(figsize=(5, 4))
        plt.title(Title)
        for i in range(self.nband):
            plt.plot(K_path, B_data[i],
                    **kwargs_plot1
                    )
            plt.scatter(
                    K_path, B_data[i],
                    c=PlotData[i],
                    **kwargs_plot2
                )
            
        plt.ylim(boundary[0], boundary[1])
        plt.xlim(np.min(K_path), np.max(K_path))
        plt.vlines(x=L_ticks, ymin=boundary[0]-1, ymax=boundary[1]+1, colors=color[-1])
        plt.hlines(y=0, xmin=np.min(K_path)-0.01, xmax=np.max(K_path)+0.01, colors=color[-1])
        plt.xticks(ticks=L_ticks, labels=L_label)
        plt.colorbar()
        plt.show()
# >>>>>>>>>> <<<<<<<<<<