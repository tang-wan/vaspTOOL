import numpy as np
import pyprocar as ppr
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import griddata

from STT_Tool import Tools
color = Tools.ColorList()


class Normal_vaspkitSTplot():
    def __init__(self):
        pass

class Project_vaspkitSTplot():
    def __init__(self):
        pass

class procarSTplot():
    def __init__(self, FilePath:str, fermiEnergy=0, PROCARtype='vasp'):
        
        self.parser   = ppr.io.Parser(code=PROCARtype, dir=FilePath)

        self.BandData    = self.parser.ebs.bands[:,:,0]-fermiEnergy
        self.kpointsData = self.parser.ebs.kpoints_cartesian
        self.kpointsData = self.kpointsData*(2*np.pi)   # Normalize to Lise's result

        self.nband   = self.parser.ebs.nbands
        self.nkpoint = self.parser.ebs.nkpoints

        Tools.Check_out_Word("#>>>>> Loading band data format <<<<<#")
        Tools.Process_Word("# =====")
        print(f"The shape of the band data: {np.shape(self.BandData)}")
        print(f"There are {self.nband} bands of each kpoint")
        print(f"There are {self.nkpoint} kpoints of each band")
    
    def Edit_interpolation_3DBand(self, wanted_E, point_num=200):
        self.wanted_E  = wanted_E
        self.point_num = point_num
        #----------------- Part 1: Check which band pass through "wanted_E" -----------------#
        inBounded_Bandindex = np.where(
                        np.logical_and(self.BandData.min(axis=0) < wanted_E, self.BandData.max(axis=0) > wanted_E)
                        )
        inBounded_Band = self.BandData.transpose()[inBounded_Bandindex[0]]
        self.inBounded_Band      = inBounded_Band
        self.inBounded_Bandindex = inBounded_Bandindex

        Tools.Check_out_Word("#>>>>> create interpolation <<<<<#")
        Tools.Process_Word("# =====")
        print(f"The band pass through number {inBounded_Bandindex[0]} bands")

        if len(inBounded_Bandindex[0]) == 0:
            # If no band pass through the wanted_E
            raise RuntimeError(f'found no bands with energy = {wanted_E}')
        
        #----------------- Part 2: Interpolate the 3D band data (x => kx, y => ky, z => band energy)  -----------------#
        kx = self.kpointsData[:,0]
        ky = self.kpointsData[:,1]

        BANDoutput_3D_Original = (kx, ky, self.BandData.transpose()[inBounded_Bandindex[0]])
        

        ##>>>>> Create the new kx, ky meshgrid with (point_num*point_num) <<<<<##
        new_kx, new_ky = np.meshgrid(np.linspace(np.min(kx), np.max(kx), point_num),
                                     np.linspace(np.min(ky), np.max(ky), point_num),
                                     indexing='ij')
        
        new_BandEnergy = np.array([])
        for Old_Band in inBounded_Band:
            ## kx, ky, and Old_Band are 1D array
            ## (kx, ky) is correspond to Old_Band
            new_BandEnergy = np.append(new_BandEnergy, 
                                       griddata(
                                           (kx, ky), Old_Band,
                                           (new_kx, new_ky), method='cubic'
                                       ))
            ## new_kx, new_ky is meshgrid array (n*n)
            ## griddata will also return the meshgrid array (n*n) which is corresponded to new_kx and new_ky
                ##>> (new_kx[0][0], new_ky[0][0]) corresponding to new_E[0][0]
        new_BandEnergy = new_BandEnergy.reshape(len(inBounded_Bandindex[0]), point_num, point_num)

        
        BANDoutput_3D = (new_kx, new_ky, new_BandEnergy)
        
        self.new_kx, self.new_ky = new_ky, new_kx
        self.new_BandEnergy      = new_BandEnergy
        
        return BANDoutput_3D_Original, BANDoutput_3D

    def Read_band_contour(self):
        wanted_E       = self.wanted_E
        new_ky, new_kx = self.new_kx, self.new_ky
        new_BandEnergy = self.new_BandEnergy
        
        Tools.Check_out_Word("#>>>>> Read band contour <<<<<#")
        Tools.Process_Word("# =====")
        #----------------- Part 3: Finding the contour of the band structure   -----------------#
        
        # Con_xData = np.array([])
        # Con_yData = np.array([])
        Con_xData, Con_yData = [], []
        Con_Array = np.array([])
        for new_Band in new_BandEnergy:
            Con = plt.contour(new_kx, new_ky, new_Band,
                            [wanted_E]
                            )
            Con_Array = np.append(Con_Array, Con)
            plt.axis("equal")
            plt.close()

            Con_xData.append(Con.get_paths()[0].vertices[:,0])
            Con_yData.append(Con.get_paths()[0].vertices[:,1])

        self.Con_xData, self.Con_yData = Con_xData, Con_yData
        self.Con_Array = Con_Array

        SToutput = (Con_xData, Con_yData)

        #----------------- Part 3-2: The format of contour line   -----------------#
        kwargs_plot1 = dict(
            color='k',
            linewidth=0.5
        )

        return SToutput, Con_Array, kwargs_plot1

    def Read_projectData(self):
        projData  = self.parser.ebs.projected[:,:,:,0,:,:]
        self.projData = projData
        # self.atomsData = projData.sum(axis=)

        self.natoms    = self.parser.ebs.natoms
        self.norbitals = self.parser.ebs.norbitals
        self.nspins    = self.parser.ebs.nspins
        
        self.atomData    = projData[:,:,:,:,0].sum(axis=3)
        self.orbitalData = projData.sum(axis=(2, 4))
        self.spinData    = projData.sum(axis=(2, 3))
        

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

        projSToutput = (self.atomData, self.orbitalData, self.spinData)

        return projSToutput
    # ==========
    def _interpolation_projectTools(self, OriginalData):

        inBounded_Bandindex  = self.inBounded_Bandindex
        inBounded_interpData = OriginalData[:,inBounded_Bandindex[0],:] # (kpoints, inBounded_Bandindex[0], 4)
        
        shape_inBounded_interpData = np.shape(inBounded_interpData)

        kx, ky = self.kpointsData[:,0], self.kpointsData[:,1]
        
        #----------------- Part 4: Interpolate the projection band data tools   -----------------#
        interpData = []
        for num_B in range(shape_inBounded_interpData[1]):
            Con_xData = self.Con_xData[num_B]
            Con_yData = self.Con_yData[num_B]

            # NewProj = np.array([])
            NewProj = []
            for num_proj in range(shape_inBounded_interpData[2]):
                OldProj  = inBounded_interpData[:, num_B, num_proj]
                gridData = griddata(
                                (kx, ky), OldProj,
                                (Con_xData, Con_yData),
                                method='cubic'
                                )
                NewProj.append(gridData)

            interpData.append(
                                np.transpose(np.c_[(*NewProj,)])
                            )
        return interpData
    # ==========
    def Edit_interpolation_projectSpin(self):
        
        OriginalData = self.spinData
        # ==========
        interpData = self._interpolation_projectTools(OriginalData)
        kwargs_spin = dict(
            s=50,
            marker='.',
            cmap='seismic',
            norm=colors.Normalize(-0.5, 0.5),
            alpha=1.0,
            edgecolor='none',
        )
              
        return interpData, kwargs_spin

    def Edit_interpolation_projectOrbital(self, orbital=["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2"]):
        
        OriginalData = self.orbitalData
        # ==========
        interpData = self._interpolation_projectTools(OriginalData)
        kwargs_orbital = dict(
            s=50,
            marker='.',
            cmap='Blues',
            norm=colors.Normalize(0, 1),
            alpha=1.0,
            edgecolor='none',
        )
              
        return interpData, kwargs_orbital

    def Edit_interpolation_projectAtom(self, AtomList:list):

        OriginalData = self.atomData[:,:,AtomList]

        OriginalData = OriginalData.sum(axis=2, keepdims=True)
        # ==========
        interpData = self._interpolation_projectTools(OriginalData)
        kwargs_atom = dict(
            s=50,
            marker='.',
            cmap='Blues',
            norm=colors.Normalize(0, 1),
            alpha=1.0,
            edgecolor='none',
        )
              
        return interpData, kwargs_atom

    def Edit_interpolation_projectAtom_comp(self, AtomList1:list, AtomList2:list, type="1-2"):

        OriginalData1 = self.atomData[:,:,AtomList1]
        OriginalData2 = self.atomData[:,:,AtomList2]

        OriginalData1 = OriginalData1.sum(axis=2, keepdims=True)
        OriginalData2 = OriginalData2.sum(axis=2, keepdims=True)

        match type:
            case "1-2":
                OriginalData = OriginalData1-OriginalData2
            case "2-1":
                OriginalData = OriginalData2-OriginalData1
            case _:
                Tools.Check_out_Word("No this kind of type")
        # ==========
        interpData = self._interpolation_projectTools(OriginalData)
        kwargs_atomcomp = dict(
            s=50,
            marker='.',
            cmap='jet',
            norm=colors.Normalize(-1, 1),
            alpha=1.0,
            edgecolor='none',
        )
              
        return interpData, kwargs_atomcomp
    # ==========
    def Plot_projectTools(self, PlotData, kwargs_plot1:dict, kwargs_plot2:dict, Setting:tuple):
        xData = self.Con_xData
        yData = self.Con_yData
        # ---------------
        plt.figure()
        plt.title(Setting[0])
        for i, pl in enumerate(PlotData):
            plt.scatter(
                x=xData[i],
                y=yData[i],
                c=pl[Setting[1]],
                **kwargs_plot2
            )
            plt.plot(xData[i],
                    yData[i], **kwargs_plot1)
        plt.colorbar()
        plt.axis("equal")
        plt.show()

# ========== Ignore ========== #
    def _LiseExample(self):
        inBounded_Bandindex  = self.inBounded_Bandindex

        point_num  = self.point_num
        kx, ky = self.kpointsData[:,0], self.kpointsData[:,1]
        
        # projData = self.parser.ebs.projected
        # spin_weights = np.sum(projData, axis=(2,3,4), keepdims=False)

        projData = self.parser.ebs.projected[:,:,:,0,:,:]
        spin_weights = projData.sum(axis=(2,3)) # (kpoints, bands, 4)

        projections = np.array([
                            spin_weights[:,:,1], # sx
                            spin_weights[:,:,2], # sy
                            spin_weights[:,:,3], # sz
                        ])
        
        projs = []
        for proj in projections:
            projs.append(proj.transpose()[inBounded_Bandindex])
        # projs will be (3, 5, 625) => (spin-dir, inbounded-band, kpoints)
        
        # ==========

        self.tables = []

        # self.Con_Array => the contour of each inbounded bands
        for i, contour in enumerate(self.Con_Array):
            for path in contour.get_paths():
                new_kx2, new_ky2 = path.vertices[:,0], path.vertices[:,1]
                # kx and ky of each contour which is in bounded band
                
                new_projs = []
                for proj in projs:
                    # proj[i] => the projection of each in bounded band
                    new_projs.append(
                                    griddata((kx, ky), proj[i], 
                                            (new_kx2, new_ky2),
                                            method='cubic'
                                            )
                                    )
                    
                table = np.c_[(new_kx2, new_ky2, *new_projs)]

                self.tables.append(table)

        return self.tables        

        

if __name__ == '__main__':
    

    def procarSTtest(plotType, protype):
        examp = procarSTplot(FilePath="testingData/1_Lise_band_novdw_soc_fermisurface_boun0.4", fermiEnergy=0.6411)
        # =====
        Old_data, data = examp.Edit_interpolation_3DBand(wanted_E=0, point_num=200)
        
        data, ConData, kwargs_plot1 = examp.Read_band_contour()
        xData, yData = data

        projData = examp.Read_projectData()
        # -----
        match plotType:
            case "spin":
                projspinData, kwargs_plot2 = examp.Edit_interpolation_projectSpin()
                # ---------------
                match protype:
                    case "sx":
                        examp.Plot_projectTools(projspinData, kwargs_plot1, kwargs_plot2, ("Spin_Sx", 1))
                    case "sy":
                        examp.Plot_projectTools(projspinData, kwargs_plot1, kwargs_plot2, ("Spin_Sy", 2))
                    case "sz":
                        examp.Plot_projectTools(projspinData, kwargs_plot1, kwargs_plot2, ("Spin_Sz", 3))
                    case _:
                        print("No this kind of spin !!!!!")

            case "Orbital":
                projOrbitalData, kwargs_plot2 = examp.Edit_interpolation_projectOrbital()
                # ---------------
                match protype:
                    case "s":
                        examp.Plot_projectTools(projOrbitalData, kwargs_plot1, kwargs_plot2, ("Orbital_s", 0))
                    case "px":
                        examp.Plot_projectTools(projOrbitalData, kwargs_plot1, kwargs_plot2, ("Orbital_px", 1))
                    case "py":
                        examp.Plot_projectTools(projOrbitalData, kwargs_plot1, kwargs_plot2, ("Orbital_py", 2))
                    case "pz":
                        examp.Plot_projectTools(projOrbitalData, kwargs_plot1, kwargs_plot2, ("Orbital_pz", 3))
                    case _:
                        print("No this kind of spin or it is out of setting !!!!!")

            case "Atom":
                # projAtomData, kwargs_plot2 = examp.Edit_interpolation_projectAtom(AtomList=[0, 1, 2, 3, 4, 5, 6])
                # ---------------
                match protype:
                    case "FM":
                        projAtomData, kwargs_plot2 = examp.Edit_interpolation_projectAtom(AtomList=[0, 1, 2, 3, 4, 5, 6])
                        examp.Plot_projectTools(projAtomData, kwargs_plot1, kwargs_plot2, ("FM", 0))
                    case "TMD":
                        projAtomData, kwargs_plot2 = examp.Edit_interpolation_projectAtom(AtomList=[7, 8, 9])
                        examp.Plot_projectTools(projAtomData, kwargs_plot1, kwargs_plot2, ("TMD", 0))

            case "Atom_comp":
                projAtomCompData, kwargs_plot2 = examp.Edit_interpolation_projectAtom_comp(AtomList1=[0, 1, 2, 3, 4, 5, 6], 
                                                                                           AtomList2=[7, 8, 9],
                                                                                           type=protype)
                # ---------------
                match protype:
                    case "1-2":
                        examp.Plot_projectTools(projAtomCompData, kwargs_plot1, kwargs_plot2, ("FM-TMD", 0))
                    case "2-1":
                        examp.Plot_projectTools(projAtomCompData, kwargs_plot1, kwargs_plot2, ("TMD-FM", 0))

# >>>>>>>>>> Runing <<<<<<<<<< #
    procarSTtest(plotType="spin", protype="sx")
    # procarSTtest(plotType="orbital", protype="pz")
    # procarSTtest(plotType="Atom", protype="FM")
    # procarSTtest(plotType="Atom_comp", protype="1-2")