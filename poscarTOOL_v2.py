import numpy as np # type: ignore
from STT_Tool import Check_out_Word as CoW
from STT_Tool import Process_Word as pw
import ase.io as ase
from collections import Counter
from itertools import groupby


def _Info_POSCARRead():
    print("=====")
    print("We have the following methods:")
    methodList = ("LayerHeight", "AtomDistance", "AtomAngle")
    for name in methodList:
        pw(f"    {name}") 
    print("=====") 
# -----
class POSCARRead():
    def __init__(self, path, p=False):
        Struct = ase.read(path)

        Formula = Struct.get_chemical_formula()
        Symbols = Struct.get_chemical_symbols()
        # print(Symbols)
        CoW(f"This POSCAR/CONTCAR is {Formula}")
        print("---")
        Vector_a1 = Struct.get_cell()[0]
        Vector_a2 = Struct.get_cell()[1]
        Vector_a3 = Struct.get_cell()[2]
        LatticeVector = np.array(
                                [Vector_a1, 
                                 Vector_a2, 
                                 Vector_a3]
                                  )
        pw("Lattice Vector:")
        print(LatticeVector)
        print("---")
        Constant_a1 = np.sqrt(Vector_a1[0]**2+Vector_a1[1]**2+Vector_a1[2]**2)
        Constant_a2 = np.sqrt(Vector_a2[0]**2+Vector_a2[1]**2+Vector_a2[2]**2)
        Constant_a3 = np.sqrt(Vector_a3[0]**2+Vector_a3[1]**2+Vector_a3[2]**2)
        pw("Lattice Constant:")
        print(Constant_a1, Constant_a2, Constant_a3)
        print("---")
        
        AtomNum = len(Struct)
        AtomPos_Cart = Struct.get_positions()
        AtomPos_Frac = Struct.get_scaled_positions()
        if p:
            pw("Atom Position (Cartesian):")
            print(AtomPos_Cart)
            print("---")
            pw("Atom Position (Fractional):")
            print(AtomPos_Frac)
            print("---")
        # ---
        self.Struct = Struct
        self.Symbols = Symbols
        self.Vector_a1, self.Vector_a2, self.Vector_a3 = Vector_a1, Vector_a2, Vector_a3
        self.Constant_a1, self.Constant_a2, self.Constant_a3 = Constant_a1, Constant_a2, Constant_a3
        self.AtomPos_Cart = AtomPos_Cart
        self.AtomPos_Frac = AtomPos_Frac

    def _ReadPos(self, TargetAtom_1, TargetAtom_2, p=(False, False)):
        Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1, self.Vector_a2, self.Vector_a3
        Constant_a1, Constant_a2, Constant_a3 = self.Constant_a1, self.Constant_a2, self.Constant_a3
        AtomPos_Cart = self.AtomPos_Cart
        AtomPos_Frac = self.AtomPos_Frac

        AtomPos_1_Frac = AtomPos_Frac[TargetAtom_1]
        AtomPos_2_Frac = AtomPos_Frac[TargetAtom_2]
        if p[0]:
            print("Atom1 Position (Fractional):")
            print(AtomPos_1_Frac)
            print("Atom2 Position:")
            print(AtomPos_2_Frac)
            print("---")

        AtomPos_1_Cart = AtomPos_Cart[TargetAtom_1]
        AtomPos_2_Cart = AtomPos_Cart[TargetAtom_2]
        if p[1]:
            print("Atom1 Position (Cartesian):")
            print(AtomPos_1_Cart)
            # print("---")
            print("Atom2 Position:")
            print(AtomPos_2_Cart)
            print("---")
        return AtomPos_1_Cart, AtomPos_2_Cart

    def LayerHeight(self):
        AtomPos_Cart = self.AtomPos_Cart

        AtomPos_z = AtomPos_Cart[:, 2]
        Sorted_AtomPos_z  = np.round(np.sort(AtomPos_z), 4)
        Sorted_AtomPos_z = [key for key, group in groupby(Sorted_AtomPos_z)]

        Sorted_AtomPos_dz = np.diff(Sorted_AtomPos_z)
        # Sorted_AtomPos_dz = Sorted_AtomPos_dz[Sorted_AtomPos_dz!=0]
        # Sorted_AtomPos_dz = [key for key, group in groupby(Sorted_AtomPos_dz)]

        pw("LayerHeight (Å):")
        print(Sorted_AtomPos_z)
        print("----")
        pw("LayerHeight difference (Å):")
        print(Sorted_AtomPos_dz)
        print("----")
    
    def AtomDistance(self, TargetAtom_1, TargetAtom_2, p=(False, False)): # Atom2 - Atom1
        self._ReadPos(TargetAtom_1, TargetAtom_2, p=p)

        AtomDist = self.Struct.get_distance(TargetAtom_1, TargetAtom_2, mic=True)

        pw(f"Atom Distance ({self.Symbols[TargetAtom_1]} - {self.Symbols[TargetAtom_2]}):")
        print(f"{AtomDist:.6f} Å")
        print("----")

        return AtomDist

    def AtomAngle(self, TargetAtom_1, TargetAtom_2, TargetAtom_3, p=(False, False, False)):
        self._ReadPos(TargetAtom_1, TargetAtom_2, p=p[0:2])
        self._ReadPos(TargetAtom_3, TargetAtom_2, p=(p[2], False))

        angle = self.Struct.get_angle(TargetAtom_1, TargetAtom_2, TargetAtom_3, mic=True)
        
        pw(f"The angle ({self.Symbols[TargetAtom_1]}-{self.Symbols[TargetAtom_2]}-{self.Symbols[TargetAtom_3]}) is:")
        print(f"{angle:.3f} degree")
        print("----")
        
        return angle

# ===========================================
def _Info_POSCAREdit():
    print("=====")
    print("We have the following methods:")
    methodList = ("SetVacCen",
                  "Centered",
                  "SuperCell",
                  "TF_repeat",
                  "LatticeStrain",
                  "LatticeSet",
                  "HeteroStructure",
                  "WritePOSCAR",
                  "Write_Atom_xyz"
                    )
    for name in methodList:
        pw(f"    {name}") 
    print("=====") 
# -----
class POSCAREdit():
    def __init__(self, path, p=False):
        Struct = ase.read(path)
        if 'momenta' in Struct.arrays:
            del Struct.arrays['momenta']

        Formula = Struct.get_chemical_formula()
        Symbols = Struct.get_chemical_symbols()
        # print(Symbols)
        CoW(f"This POSCAR/CONTCAR is {Formula}")
        print("---")
        Vector_a1 = Struct.get_cell()[0]
        Vector_a2 = Struct.get_cell()[1]
        Vector_a3 = Struct.get_cell()[2]
        LatticeVector = np.array(
                                [Vector_a1, 
                                Vector_a2, 
                                Vector_a3]
                                )
        # print("Lattice Vector:")
        # print(LatticeVector)
        # print("---")
        Constant_a1 = np.sqrt(Vector_a1[0]**2+Vector_a1[1]**2+Vector_a1[2]**2)
        Constant_a2 = np.sqrt(Vector_a2[0]**2+Vector_a2[1]**2+Vector_a2[2]**2)
        Constant_a3 = np.sqrt(Vector_a3[0]**2+Vector_a3[1]**2+Vector_a3[2]**2)
        # print("Lattice Constant:")
        # print(Constant_a1, Constant_a2, Constant_a3)
        # print("---")
        
        AtomNum = len(Struct)
        AtomPos_Cart = Struct.get_positions()
        AtomPos_Frac = Struct.get_scaled_positions()
        # if p:
        #     print("Atom Position (Cartesian):")
        #     print(AtomPos_Cart)
        #     print("---")
        #     print("Atom Position (Fractional):")
        #     print(AtomPos_Frac)
        #     print("---")
        # ---
        self.Struct = Struct
        self.Symbols = Symbols
        self.LatticeVector = LatticeVector
        self.Vector_a1, self.Vector_a2, self.Vector_a3 = Vector_a1, Vector_a2, Vector_a3
        self.Constant_a1, self.Constant_a2, self.Constant_a3 = Constant_a1, Constant_a2, Constant_a3
        self.AtomPos_Cart = AtomPos_Cart
        self.AtomPos_Frac = AtomPos_Frac

    def SetVacCen(self, Vac=15.0, axis=2):
        self.Struct.center(vacuum=Vac, axis=axis)

    def Centered(self, axis=2):
        self.Struct.center(axis=axis)

    def SuperCell(self, repeatCell:tuple):
        self.Struct = self.Struct*repeatCell

    def TF_repeat(self, thickness:float, vdw:float, N=2):
    #! Only 1-TL to n-TL
        base_struct = self.Struct.copy()
        vdwStruct = base_struct.copy()
        
        dStruct = thickness + vdw
        for n in range(1, N):
            ShiftStruct = base_struct.copy()
            ShiftStruct.translate([0, 0, dStruct * n])
            vdwStruct = vdwStruct + ShiftStruct

        self.Struct = vdwStruct

    def LatticeStrain(self, StrainMatrix:tuple, atomScale=True):
        Vector_a1 = self.Struct.get_cell()[0]
        Vector_a2 = self.Struct.get_cell()[1]
        Vector_a3 = self.Struct.get_cell()[2]
        LatticeVector = np.array(
                                [Vector_a1, 
                                Vector_a2, 
                                Vector_a3]
                                )
        
        new_LatticeVector = LatticeVector.copy()
        new_LatticeVector[0]*=StrainMatrix[0]
        new_LatticeVector[1]*=StrainMatrix[1]
        new_LatticeVector[2]*=StrainMatrix[2]
        self.Struct.set_cell(new_LatticeVector, scale_atoms=atomScale)
        # self.Struct.center(axis=2)

    def LatticeSet(self, LatticeMatrix:np.array, atomScale=False):
        self.Struct.set_cell([LatticeMatrix[0], LatticeMatrix[1], LatticeMatrix[2]], scale_atoms=atomScale)
        # self.Struct.center(axis=2)

    def HeteroStructure(self, top_path, vdw=3.20, Vac=15.0):
        bottom_layer = self.Struct.copy()
        top_layer = ase.read(top_path)
        
        if 'momenta' in top_layer.arrays:
            del top_layer.arrays['momenta']

        new_top_cell = top_layer.get_cell().copy()
        new_top_cell[0] = bottom_layer.get_cell()[0]
        new_top_cell[1] = bottom_layer.get_cell()[1]
        top_layer.set_cell(new_top_cell, scale_atoms=True)

        bottom_z_max = bottom_layer.positions[:, 2].max()
        top_z_min = top_layer.positions[:, 2].min()
        shift_z = (bottom_z_max + vdw) - top_z_min
        top_layer.translate([0, 0, shift_z])

        hetero = bottom_layer + top_layer
        hetero.center(vacuum=Vac, axis=2)
        
        self.Struct = hetero
        pw(f"Heterostructure built! Top layer loaded from:")
        print(f"     {top_path}")
        print("---")
# ==========================
    def _get_sorted_struct(self):
        unique_symbols = []
        for sym in self.Struct.get_chemical_symbols():
            if sym not in unique_symbols:
                unique_symbols.append(sym)

        def sort_key(index):
            atom = self.Struct[index]
            return (unique_symbols.index(atom.symbol), atom.position[2])

        sorted_indices = sorted(range(len(self.Struct)), key=sort_key)

        return self.Struct[sorted_indices]
    
    def WritePOSCAR(self, name):
        sorted_supercell = self._get_sorted_struct()
        ase.write(name, sorted_supercell, 
                format='vasp', 
                vasp5=True, 
                direct=True)
        pw(f"Writing POSCAR Finished:")
        print(f"    {name}")
        print("=====")

    def Write_Atom_xyz(self, Path, r, theta, phi):            
        # 使用已經完美排序過的結構
        sorted_struct = self._get_sorted_struct()
        AtomType = sorted_struct.get_chemical_symbols()
        Cart_AtomPos = sorted_struct.get_positions()
        
        # 依序計算每種元素的數量 (避免 np.unique 亂排字母順序)
        # 例如: ['Bi', 'Bi', 'Se', 'Se'] -> {'Bi': 2, 'Se': 2}
        counts_dict = Counter(AtomType)
        # 保持原本的元素順序
        unique_symbols = list(dict.fromkeys(AtomType)) 

        pw("Start to convert vasp POSCAR to nanodcal atom.xyz")
        
        with open(Path, "w") as W_File:
            W_File.write(f"{len(Cart_AtomPos)}\n")
            W_File.write(f"{'AtomType':<8}{'X':^20}{'Y':^20}{'Z':^20}{'SpinPolarization_r':^30}{'SpinPolarization_theta':^30}{'SpinPolarization_phi':^30}\n")
            
            pos = 0
            for i, sym in enumerate(unique_symbols):
                rep = counts_dict[sym]
                print(f"Writing {rep} atoms of {sym}")
                for _ in range(rep):
                    atom = AtomType[pos]
                    W_File.write(f"{atom:<8}{Cart_AtomPos[pos][0]:>20.14f}{Cart_AtomPos[pos][1]:>20.14f}{Cart_AtomPos[pos][2]:>20.14f}{r[i]:^30.14f}{theta[i]:^30.14f}{phi[i]:^30.14f}\n")
                    pos += 1
            
        pw("Converting Finished.")
        print("=====")
# =====