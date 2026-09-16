import numpy as np
import pyprocar as ppr
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import griddata

from STT_Tool import Process_Word as pw
from STT_Tool import Tools

color = Tools.ColorList()

# >>>>>>>>>> Info Block <<<<<<<<<< #
def _Info_():
    print("=====")
    print("We have the following methods:")
    methodList = (
        "Edit_interpolation_3DBand",
        "Read_band_contour",
        "Read_SpinData_projectionData",
        "Read_OrbitalData_projectionData",
        "Read_AtomData_projectionData",
        "Read_AtomCompData_projectionData",
        "kwargsList"
    )
    for name in methodList:
        pw(f"    {name}") 
    print("=====") 
# >>>>>>>>>> <<<<<<<<<< #

class procarSTplot():
    def __init__(self, FilePath:str, fermiEnergy=0, PROCARtype='vasp'):
        
        print(">>>>> pyprocar version <<<<<")
        print(ppr.__version__)
        print(">>>>> =============== <<<<<")
        print()
        
        self.parser = ppr.io.Parser(code=PROCARtype, dirpath=FilePath)

        self.BandData    = self.parser.ebs.bands[:, :, 0] - fermiEnergy
        self.kpointsData = self.parser.ebs.kpoints_cartesian
        self.kpointsData = self.kpointsData * (2 * np.pi)   # Normalize to Lise's result

        self.nband   = self.parser.ebs.nbands
        self.nkpoint = self.parser.ebs.nkpoints
        
        self.table = """
+-------+-----+------+------+------+------+------+------+------+------+
|n-lm   |  0  |   1  |  2   |   3  |   4  |   5  |   6  |   7  |   8  |
+=======+=====+======+======+======+======+======+======+======+======+
|-1(tot)|  s  |  py  |  pz  |  px  | dxy  | dyz  | dz2  | dxz  |x2-y2 |
+-------+-----+------+------+------+------+------+------+------+------+
"""

        Tools.Check_out_Word("#>>>>> Loading band data format <<<<<#")
        Tools.Process_Word("# =====")
        print(f"The shape of the band data: {np.shape(self.BandData)}")
        print(f"There are {self.nband} bands of each kpoint")
        print(f"There are {self.nkpoint} kpoints of each band")

        # ----------------- 宣告所有畫圖設定為實例屬性 (Instance Attributes) ----------------- #
        self.kwargs_line = dict(
            color='k',
            linewidth=0.5
        )
        
        self.kwargs_seismic = dict(
            s=50,
            marker='.',
            cmap='seismic',
            norm=colors.Normalize(-0.5, 0.5),
            alpha=1.0,
            edgecolor='none',
        )
        
        self.kwargs_blue = dict(
            s=50,
            marker='.',
            cmap='Blues',
            norm=colors.Normalize(0, 1),
            alpha=1.0,
            edgecolor='none',
        )
        
        self.kwargs_jet = dict(
            s=50,
            marker='.',
            cmap='jet',
            norm=colors.Normalize(-1, 1),
            alpha=1.0,
            edgecolor='none',
        )
    
# ==========
    def Edit_interpolation_3DBand(self, wanted_E, point_num=200):
        self.wanted_E  = wanted_E
        self.point_num = point_num
        
        # ----------------- Part 1: Check which band passes through "wanted_E" ----------------- #
        inBounded_Bandindex = np.where(
                        np.logical_and(self.BandData.min(axis=0) < wanted_E, self.BandData.max(axis=0) > wanted_E)
                        )
        inBounded_Band = self.BandData.transpose()[inBounded_Bandindex[0]]
        self.inBounded_Band      = inBounded_Band
        self.inBounded_Bandindex = inBounded_Bandindex

        Tools.Check_out_Word("#>>>>> create interpolation <<<<<#")
        Tools.Process_Word("# =====")
        print(f"The band passes through {len(inBounded_Bandindex[0])} bands: {inBounded_Bandindex[0]}")

        if len(inBounded_Bandindex[0]) == 0:
            raise RuntimeError(f'Found no bands with energy = {wanted_E}')
        
        # ----------------- Part 2: Interpolate the 3D band data ----------------- #
        kx = self.kpointsData[:, 0]
        ky = self.kpointsData[:, 1]

        BANDoutput_3D_Original = (kx, ky, self.BandData.transpose()[inBounded_Bandindex[0]])
        
        new_kx, new_ky = np.meshgrid(np.linspace(np.min(kx), np.max(kx), point_num),
                                     np.linspace(np.min(ky), np.max(ky), point_num),
                                     indexing='ij')
        
        new_BandEnergy = np.array([])
        for Old_Band in inBounded_Band:
            new_BandEnergy = np.append(new_BandEnergy, 
                                       griddata((kx, ky), Old_Band, (new_kx, new_ky), method='cubic'))
                                       
        new_BandEnergy = new_BandEnergy.reshape(len(inBounded_Bandindex[0]), point_num, point_num)

        BANDoutput_3D = (new_kx, new_ky, new_BandEnergy)
        
        self.new_kx, self.new_ky = new_ky, new_kx
        self.new_BandEnergy      = new_BandEnergy
        
        return BANDoutput_3D_Original, BANDoutput_3D

# ==========
    def Read_band_contour(self):
        wanted_E       = self.wanted_E
        new_ky, new_kx = self.new_kx, self.new_ky
        new_BandEnergy = self.new_BandEnergy
        
        Tools.Check_out_Word("#>>>>> Read band contour <<<<<#")
        Tools.Process_Word("# =====")
        
        # ----------------- Part 3: Finding the contour of the band structure ----------------- #
        Con_xData, Con_yData = [], []
        Con_Array = np.array([])
        for new_Band in new_BandEnergy:
            Con = plt.contour(new_kx, new_ky, new_Band, [wanted_E])
            Con_Array = np.append(Con_Array, Con)
            plt.axis("equal")
            plt.close()

            Con_xData.append(Con.get_paths()[0].vertices[:, 0])
            Con_yData.append(Con.get_paths()[0].vertices[:, 1])

        self.Con_xData, self.Con_yData = Con_xData, Con_yData
        self.Con_Array = Con_Array

        SToutput = (Con_xData, Con_yData)

        
        kwargs_plot1 = self.kwargs_line

        return SToutput, Con_Array, kwargs_plot1

# ==========
    def _interpolate_1D_projectData(self, projData_1D):
        inBounded_interpData = projData_1D[:, self.inBounded_Bandindex[0]]
        
        kx = self.kpointsData[:, 0]
        ky = self.kpointsData[:, 1]
        
        # ----------------- Part 4: Interpolate the projection band data tools ----------------- #
        interpData = []
        for num_B in range(inBounded_interpData.shape[1]):
            Con_xData = self.Con_xData[num_B]
            Con_yData = self.Con_yData[num_B]

            OldProj  = inBounded_interpData[:, num_B]
            gridData = griddata((kx, ky), OldProj, (Con_xData, Con_yData), method='cubic')
            interpData.append(gridData)
            
        return interpData

# ==========
    def kwargsList(self):
        return self.kwargs_line, self.kwargs_seismic, self.kwargs_blue, self.kwargs_jet

# ==========
    def Read_SpinData_projectionData(self, spinList=(0,)):
        spinData = self.parser.ebs.ebs_sum(atoms=None, orbitals=None, spins=spinList)
        interpData = self._interpolate_1D_projectData(spinData)
        
        kwargs_spin = self.kwargs_seismic
        return interpData, kwargs_spin

# ==========
    def Read_OrbitalData_projectionData(self, orbitalList=None, table=0):
        orbitalData = self.parser.ebs.ebs_sum(atoms=None, orbitals=orbitalList, spins=(0,))
        interpData = self._interpolate_1D_projectData(orbitalData)
        
        kwargs_orbital = self.kwargs_blue
        if table:
            print(self.table)
            
        return interpData, kwargs_orbital

# ==========
    def Read_AtomData_projectionData(self, atomList=None):
        atomData = self.parser.ebs.ebs_sum(atoms=atomList, orbitals=None, spins=(0,))
        interpData = self._interpolate_1D_projectData(atomData)
        
        kwargs_atom = self.kwargs_blue
        return interpData, kwargs_atom

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
            
        interpData = self._interpolate_1D_projectData(atomData)
        
        
        kwargs_atomcomp = self.kwargs_jet
        return interpData, kwargs_atomcomp

# ==========
    def Plot_projectTools(self, PlotData, kwargs_plot1:dict, kwargs_plot2:dict, Title:str, 
                          yboundary=None, xboundary=None):
        xData = self.Con_xData
        yData = self.Con_yData
        
        # ---------------
        plt.figure(figsize=(5, 5))
        plt.title(Title)
        
        for i, proj in enumerate(PlotData):
            plt.plot(xData[i], yData[i], **kwargs_plot1)
            plt.scatter(xData[i], yData[i], c=proj, **kwargs_plot2)
            
        plt.axis("equal")
        
        if xboundary is not None:
            plt.xlim(xboundary[0], xboundary[1])
        if yboundary is not None:
            plt.ylim(yboundary[0], yboundary[1])
            
        plt.colorbar()
        plt.show()

