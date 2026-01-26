import numpy as np # type: ignore
from STT_Tool import Check_out_Word as CoW
from STT_Tool import Tools as st


class POSCARRead():
    def __init__(self, path):
        with open(path, "r") as R_File:
            dataALL = R_File.readlines()
        Const = np.float64(dataALL[1])
        Vector_a1 = np.float64(dataALL[2].split())*Const
        Vector_a2 = np.float64(dataALL[3].split())*Const
        Vector_a3 = np.float64(dataALL[4].split())*Const
        LatticeVector = np.array([Vector_a1, 
                                  Vector_a2, 
                                  Vector_a3])
        # LatticeConstant_a1 = np.sqrt(np.sum(Vector_a1**2))
        LatticeConstant_a1 = st.Length(Vector_a1)
        LatticeConstant_a2 = st.Length(Vector_a2)
        LatticeConstant_a3 = st.Length(Vector_a3)
        print("Lattice Constant:")
        print(LatticeConstant_a1, LatticeConstant_a2, LatticeConstant_a3)
        print("---")
        print("Lattice Vector:")
        print(LatticeVector)
        print("=====")
        AtomNum = np.int32(dataALL[6].split())
        AtomPos = np.array([])
        for a in range(sum(AtomNum)):
            try:
                AtomPos = np.append(AtomPos, np.float64(dataALL[8+a].split()))
            except ValueError:
                AtomPos = np.append(AtomPos, np.float64(dataALL[8+a].split()[:-1]))
        AtomPos = AtomPos.reshape(sum(AtomNum), 3)
        print("Atom Position:")
        print(AtomPos)
        # ---
        self.Vector_a1, self.Vector_a2, self.Vector_a3 = Vector_a1, Vector_a2, Vector_a3
        self.LatticeConstant_a1, self.LatticeConstant_a2, self.LatticeConstant_a3 = LatticeConstant_a1, LatticeConstant_a2, LatticeConstant_a3
        self.AtomPos = AtomPos

    def _ReadPos(self, TargetAtom_1, TargetAtom_2, p=(True, True)):
        Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1, self.Vector_a2, self.Vector_a3
        LatticeConstant_a1, LatticeConstant_a2, LatticeConstant_a3 = self.LatticeConstant_a1, self.LatticeConstant_a2, self.LatticeConstant_a3
        AtomPos = self.AtomPos

        AtomPos_1_Frac = AtomPos[TargetAtom_1]
        AtomPos_2_Frac = AtomPos[TargetAtom_2]
        if p[0]:
            print("Atom1 Position (Fractional):")
            print(AtomPos_1_Frac)
            # print("---")
            print("Atom2 Position:")
            print(AtomPos_2_Frac)
            print("---")
        AtomPos_1_Cart = AtomPos_1_Frac[0]*Vector_a1 + AtomPos_1_Frac[1]*Vector_a2 + AtomPos_1_Frac[2]*Vector_a3
        AtomPos_2_Cart = AtomPos_2_Frac[0]*Vector_a1 + AtomPos_2_Frac[1]*Vector_a2 + AtomPos_2_Frac[2]*Vector_a3
        if p[1]:
            print("Atom1 Position (Cartesian):")
            print(AtomPos_1_Cart)
            # print("---")
            print("Atom2 Position:")
            print(AtomPos_2_Cart)
            print("---")

        return AtomPos_1_Cart, AtomPos_2_Cart

    def LayerHeight(self, TargetAtom_1, TargetAtom_2, p=(True, True)): # Atom2 - Atom1
        AtomPos_1_Cart, AtomPos_2_Cart = self._ReadPos(TargetAtom_1, TargetAtom_2, p=p)
        
        speczdiff = AtomPos_2_Cart[2]-AtomPos_1_Cart[2]
        print("LayerHeight:")
        print(speczdiff)
        return speczdiff
    
    def AtomDistance(self, TargetAtom_1, TargetAtom_2, p=(True, True)): # Atom2 - Atom1
        AtomPos_1_Cart, AtomPos_2_Cart = self._ReadPos(TargetAtom_1, TargetAtom_2, p=p)

        AtomDist_Cart = AtomPos_2_Cart - AtomPos_1_Cart
        # specDist = np.sqrt(np.sum(AtomDist_Cart**2))
        specDist = st.Length(AtomDist_Cart)
        print("Atom2 - Atom1:")
        print(AtomDist_Cart)
        # print("---")
        print("Distance:")
        print(specDist, "Å")
        return specDist

    def AtomAngle(self, TargetAtom_1, TargetAtom_2, TargetAtom_3, p=(True, True, True)):
        AtomPos_1_Cart, AtomPos_2_Cart = self._ReadPos(TargetAtom_1, TargetAtom_2, p=p[0:2])
        AtomPos_3_Cart, _ = self._ReadPos(TargetAtom_3, TargetAtom_2, p=(p[2], False))

        Vec1 = AtomPos_1_Cart - AtomPos_2_Cart
        Vec2 = AtomPos_3_Cart - AtomPos_2_Cart

        angle = (Vec1@Vec2)/(st.Length(Vec1)*st.Length(Vec2))
        angle = np.arccos(angle)*180/np.pi
        
        print("The angle is:")
        print(angle, "degree")

    def CrossSection_Rec(self, EndpointAtom_1, EndpointAtom_2, p=(True, True)):
        EndpointAtom_1_Cart, EndpointAtom_2_Cart = self._ReadPos(EndpointAtom_1, EndpointAtom_2, p=p)

        width  = np.sqrt(
                (EndpointAtom_2_Cart[0] - EndpointAtom_1_Cart[0])**2+(EndpointAtom_2_Cart[1]-EndpointAtom_1_Cart[1])**2
                )
        
        height = np.abs(EndpointAtom_2_Cart[2] - EndpointAtom_1_Cart[2])
        Area = width*height 

        print("Cross Section Area:")
        print(Area, "Å^2")

        # print("Repairing...")

        return Area

    def CrossSection_Hex(self, EndpointAtom_1_Cart, EndpointAtom_2_Cart, EndpointAtom_3_Cart, p=(True, True, True)):
        # EndpointAtom_1_Cart, EndpointAtom_2_Cart = self._ReadPos(EndpointAtom_1, EndpointAtom_2, p=p[0:2])
        # EndpointAtom_3_Cart, _ = self._ReadPos(EndpointAtom_3, EndpointAtom_2, p=(p[2], False))
        Dis12 = st.Length(EndpointAtom_1_Cart-EndpointAtom_2_Cart)
        Dis23 = st.Length(EndpointAtom_2_Cart-EndpointAtom_3_Cart)
        Dis31 = st.Length(EndpointAtom_3_Cart-EndpointAtom_1_Cart)

        if np.abs(Dis12 - Dis23)>1e-3 or np.abs(Dis23 - Dis31)>1e-3:
            CoW("Warning!! This is not a regular hexagon.")
        if EndpointAtom_1_Cart[-1]-EndpointAtom_2_Cart[-1]>1e-3 or EndpointAtom_2_Cart[-1]-EndpointAtom_3_Cart[-1]>1e-3:
            CoW("Warning!! The three atoms are not in the same z-plane.")

        Dis = np.mean((Dis12, Dis23, Dis31))
        TriArea = (np.sqrt(3)/4)*Dis**2
        Area = TriArea*2
        
        print("Cross Section Area:")
        print(Area, "Å^2")

        return Area

class POSCARConvert():
    def __init__(self, path, p=True):
        with open(path, "r") as R_File:
            dataALL = R_File.readlines()
        Const = np.float64(dataALL[1])
        Vector_a1 = np.float64(dataALL[2].split())*Const
        Vector_a2 = np.float64(dataALL[3].split())*Const
        Vector_a3 = np.float64(dataALL[4].split())*Const
        LatticeVector = np.array([Vector_a1, 
                                Vector_a2, 
                                Vector_a3])
        LatticeConstant_a1 = np.sqrt(np.sum(Vector_a1*Vector_a1))
        LatticeConstant_a2 = np.sqrt(np.sum(Vector_a2*Vector_a2))
        LatticeConstant_a3 = np.sqrt(np.sum(Vector_a3*Vector_a3))
        if p:
            print("Lattice Constant:")
            print(LatticeConstant_a1, LatticeConstant_a2, LatticeConstant_a3)
            print("---")
            print("Lattice Vector:")
            print(LatticeVector)
            print("=====")
        AtomType = dataALL[5].split()
        AtomNum  = np.int32(dataALL[6].split())
        AtomPos  = np.array([])
        for a in range(sum(AtomNum)):
            try:
                AtomPos = np.append(AtomPos, np.float64(dataALL[8+a].split()))
            except ValueError:
                AtomPos = np.append(AtomPos, np.float64(dataALL[8+a].split()[:-1]))
        AtomPos = AtomPos.reshape(sum(AtomNum), 3)

        if p:
            print("Atom Position:")
            print(AtomPos)
        # ---
        self.Vector_a1, self.Vector_a2, self.Vector_a3 = Vector_a1, Vector_a2, Vector_a3
        self.LatticeConstant_a1, self.LatticeConstant_a2, self.LatticeConstant_a3 = LatticeConstant_a1, LatticeConstant_a2, LatticeConstant_a3
        self.AtomType = AtomType
        self.AtomNum  = AtomNum
        self.AtomPos  = AtomPos
  
    def AddVac(self, Vac, p=(True, True)):
        try:
            Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1_modified, self.Vector_a2_modified, self.Vector_a3_modified
            AtomPos = self.AtomPos_modified
            print(">> Not New Start <<")
        except AttributeError:
            Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1, self.Vector_a2, self.Vector_a3
            AtomPos = self.AtomPos
            print(">> New Start <<")

        Frac_Car_Matrix = np.array([Vector_a1,
                                    Vector_a2,
                                    Vector_a3]
                                    )
        Cart_AtomPos = AtomPos@Frac_Car_Matrix
        
        Vac_Matrix = np.zeros_like(Cart_AtomPos)
        Vac_Matrix[:,-1] = Vac/2

        Vaced_Cart_AtomPos = Cart_AtomPos + Vac_Matrix
        
        Vaced_Vector_a3 = Vector_a3 + np.array([0, 0, Vac])
        Vaced_Vector_a2 = Vector_a2
        Vaced_Vector_a1 = Vector_a1
        
        Vaced_Frac_Car_Matrix = np.array([Vaced_Vector_a1,
                                          Vaced_Vector_a2,
                                          Vaced_Vector_a3]
                                          )
        Vaced_AtomPos = Vaced_Cart_AtomPos@np.linalg.inv(Vaced_Frac_Car_Matrix)

        if p[0]:
            print("Lattice Vector (Before):")
            print(Frac_Car_Matrix)
            print("-----")
            print(AtomPos)
            print()
        if p[1]:
            print("Lattice Vector (After):")
            print(Vaced_Frac_Car_Matrix)
            print("-----")
            print(Vaced_AtomPos)

        self.Vector_a1_modified, self.Vector_a2_modified, self.Vector_a3_modified = Vaced_Vector_a1, Vaced_Vector_a2, Vaced_Vector_a3
        self.AtomPos_modified = Vaced_AtomPos
    
    def DelAtom(self, delpos:tuple, height=None, p=(True, True)):      
        try:
            Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1_modified, self.Vector_a2_modified, self.Vector_a3_modified
            AtomPos = self.AtomPos_modified
            print(">> Not New Start <<")
        except AttributeError:
            Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1, self.Vector_a2, self.Vector_a3
            AtomPos = self.AtomPos
            print(">> New Start <<")
        
        Frac_Car_Matrix = np.array([Vector_a1,
                                    Vector_a2,
                                    Vector_a3]
                                    )
        
        heightRatio = height/Vector_a3[-1]

        Deled_AtomPos = np.delete(AtomPos, delpos, axis=0)
        Deled_Vector_a1 = Vector_a1
        Deled_Vector_a2 = Vector_a2
        Vector_a3[-1] = Vector_a3[-1]*heightRatio
        Deled_Vector_a3 = Vector_a3

        Deled_Frac_Car_Matrix = np.array([Deled_Vector_a1,
                                          Deled_Vector_a2,
                                          Deled_Vector_a3]
                                          )

        Deled_AtomPos[:,2] = Deled_AtomPos[:,2]*(1/heightRatio)

        if p[0]:
            print("Lattice Vector (Before):")
            print(Frac_Car_Matrix)
            print("-----")
            print(AtomPos)
            print()
        if p[1]:
            print("Lattice Vector (After):")
            print(Deled_Frac_Car_Matrix)
            print("-----")
            print(Deled_AtomPos)

        self.Vector_a1_modified, self.Vector_a2_modified, self.Vector_a3_modified = Deled_Vector_a1, Deled_Vector_a2, Deled_Vector_a3
        self.AtomPos_modified = Deled_AtomPos

    def ShiftCent(self, p=(True, True)):
        try:
            Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1_modified, self.Vector_a2_modified, self.Vector_a3_modified
            AtomPos = self.AtomPos_modified
            print(">> Not New Start <<")
        except AttributeError:
            Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1, self.Vector_a2, self.Vector_a3
            AtomPos = self.AtomPos
            print(">> New Start <<") 
        
        Frac_Car_Matrix = np.array([Vector_a1,
                                    Vector_a2,
                                    Vector_a3]
                                    )

        AveragePos = np.mean(AtomPos[:,2])
        deltaAverage = AveragePos - 0.5

        Shift_Matrix = np.zeros_like(AtomPos)
        Shift_Matrix[:,-1] = -deltaAverage

        Shifted_AtomPos = AtomPos + Shift_Matrix
        Shifted_Vector_a1 = Vector_a1
        Shifted_Vector_a2 = Vector_a2
        Shifted_Vector_a3 = Vector_a3

        if p[0]:
            print("Lattice Vector (Before):")
            print(Frac_Car_Matrix)
            print("-----")
            print(AtomPos)
            print()
        if p[1]:
            print("Lattice Vector (After):")
            print(Frac_Car_Matrix)
            print("-----")
            print(Shifted_AtomPos)

        self.Vector_a1_modified, self.Vector_a2_modified, self.Vector_a3_modified = Shifted_Vector_a1, Shifted_Vector_a2, Shifted_Vector_a3
        self.AtomPos_modified = Shifted_AtomPos

    #TODO: Not finished, still worling......
    def RepeatCell_z(self, repZ, vdwGap=6.68915):
        try:
            Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1_modified, self.Vector_a2_modified, self.Vector_a3_modified
            AtomPos = self.AtomPos_modified
        except AttributeError:
            Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1, self.Vector_a2, self.Vector_a3
            AtomPos = self.AtomPos
        
        Frac_Car_Matrix = np.array([Vector_a1,
                                    Vector_a2,
                                    Vector_a3]
                                    )
        Cart_AtomPos = AtomPos@Frac_Car_Matrix

    def WritePOSCAR(self, outPath, atomTypes, atomNums):
        Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1_modified, self.Vector_a2_modified, self.Vector_a3_modified
        AtomPos = self.AtomPos_modified
        with open(outPath, "w") as W_File:
            W_File.write("Modified POSCAR\n")
            W_File.write("1.0\n")
            W_File.write(f"   {Vector_a1[0]:>.16f}   {Vector_a1[1]:>.16f}   {Vector_a1[2]:>.16f}\n")
            W_File.write(f"   {Vector_a2[0]:>.16f}   {Vector_a2[1]:>.16f}   {Vector_a2[2]:>.16f}\n")
            W_File.write(f"   {Vector_a3[0]:>.16f}   {Vector_a3[1]:>.16f}   {Vector_a3[2]:>.16f}\n")
            W_File.write(f"{atomTypes}\n")
            W_File.write(f"{atomNums}\n")
            W_File.write("Direct\n")
            for pos in AtomPos:
                W_File.write(f"   {pos[0]:.16f}   {pos[1]:.16f}   {pos[2]:.16f}\n")

    def Write_Atom_xyz(self, outPath, r, theta, phi, p=(True, True)):
        try:
            Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1_modified, self.Vector_a2_modified, self.Vector_a3_modified
            AtomPos = self.AtomPos_modified
        except AttributeError:
            Vector_a1, Vector_a2, Vector_a3 = self.Vector_a1, self.Vector_a2, self.Vector_a3
            AtomPos = self.AtomPos
        
        AtomType = self.AtomType
        AtomNum  = self.AtomNum

        print("Start to convert vasp POSCAR to nanodcal atom.xyz")
        
        Frac_Car_Matrix = np.array([Vector_a1,
                                    Vector_a2,
                                    Vector_a3]
                                    )
        Cart_AtomPos = AtomPos@Frac_Car_Matrix
        with open(outPath, "w") as W_File:
            W_File.write(f"{len(AtomPos)}\n")
            W_File.write(f"{'AtomType':<8}{'X':^20}{'Y':^20}{'Z':^20}{'SpinPolarization_r':^30}{'SpinPolarization_theta':^30}{'SpinPolarization_phi':^30}\n")
            pos = 0
            for i, rep in enumerate(AtomNum):
                atom = AtomType[i]
                for _ in range(rep):
                    W_File.write(f"{atom:<8}{Cart_AtomPos[pos][0]:>20.14f}{Cart_AtomPos[pos][1]:>20.14f}{Cart_AtomPos[pos][2]:>20.14f}{r[i]:^30.14f}{theta[i]:^30.14f}{phi[i]:^30.14f}\n")
                    pos += 1
        
        print("Converting Finished.")