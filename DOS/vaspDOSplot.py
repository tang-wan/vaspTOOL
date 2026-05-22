import numpy as np
import pyprocar as ppr

from STT_Tool import Tools
color = Tools.ColorList()

class procarDOSplot():
    def __init__(self, FilePath:str, PROCARtype='vasp'):

        print(">>>>> pyprocar verision <<<<<")
        print(ppr.__version__)
        print(">>>>> =============== <<<<<")
        print()
        
        parser = ppr.io.Parser(code=PROCARtype, dirpath=FilePath)
        self.parser = parser.dos
        
        efermi = self.parser.efermi
        self.efermi = efermi
        self.Edata = self.parser.energies - efermi
    
        self.table = """
+-------+-----+------+------+------+------+------+------+------+------+
|n-lm   |  0  |   1  |  2   |   3  |   4  |   5  |   6  |   7  |   8  |
+=======+=====+======+======+======+======+======+======+======+======+
|-1(tot)|  s  |  py  |  pz  |  px  | dxy  | dyz  | dz2  | dxz  |x2-y2 |
+-------+-----+------+------+------+------+------+------+------+------+
"""

    def Read_AllData_DOS(self):
        parser = self.parser

        DOSdata = parser.total
        Edata   = self.Edata

        Tools.Process_Word("# >>>>>>>>>> Density of States <<<<<<<<<< #")
        print("# ==========")
        print("(spin, density)")
        print(np.shape(DOSdata))
        print("# -----")
        print("(energy)")
        print(np.shape(Edata))
        print("# -----")
        print("Fermi Energy")
        print(self.efermi)
        print("# ==========")

        kwargs_plotup = dict(
            color='red',
            linewidth=1.0
        )

        kwargs_plotdn = dict(
            color='blue',
            linewidth=1.0
        )

        DOSoutput  = (Edata, DOSdata)
        kwargs_plot = (kwargs_plotup, kwargs_plotdn)

        return DOSoutput, kwargs_plot
    
    def Read_AllData_projectionData(self, atomList=(0,), orbitList=(4, 5, 6, 7, 8), spinsList=None, table=0):
        parser, Edata = self.parser, self.Edata

        projData = parser.projected[:,0,:,:,:]
        
        natoms = np.shape(projData)[0]
        norbitals = np.shape(projData)[1]
        nspins = np.shape(projData)[2]

        self.atomData    = projData.sum(axis=1)
        self.orbitalData = projData.sum(axis=0)
        
        if table:
            Tools.Check_out_Word("#>>>>> Read project DOS data <<<<<#")
            print(f"The shape of the projection data: {np.shape(projData)}")
            print(f"There are {natoms} atoms in this data")
            print(f"There are {norbitals} orbitals in this data")
            print(f"There are {nspins} spins in this data")
            print(self.table)

        if nspins == 4:
            print(f"[0 => Total DOS; 1 => Sx; 2 => Sy; 3 => Sz]")
        elif nspins == 2:
            print(f"[0 => spin up; 1 => spin down]")

        projData = parser.dos_sum(atoms=atomList, orbitals=orbitList, spins=spinsList)
        DOSoutput = (Edata, projData)

        return DOSoutput