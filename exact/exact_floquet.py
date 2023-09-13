import numpy as np
from scipy import linalg

class Floquet_Ising_ED:

    def __init__(self, model_params):
        self.Lx = model_params['Lx']
        self.Ly = model_params['Ly']
        self.bc = model_params.get('bc', [0, 0])
        self.T = model_params.get('T', 1.0)
        self.g = model_params.get('g', 1.0)
        self.J = model_params.get('J', 1.0)
        self.init_matrices()
        self.init_bond_indices()
        self.init_up_state()

    def init_matrices(self):
        """
        initialize the Floquet time evolution matrices
        """
        X = np.array([[0., 1.], [1., 0.]])
        Z = np.array([[1., 0.], [0., -1.]])
        self.X = np.matrix(X)
        self.Z = np.matrix(Z)
        self.H_X = - self.J * self.g * self.X
        self.H_ZZ = - self.J * np.kron(Z, Z).reshape([2, 2, 2, 2])
        self.U_X = linalg.expm(-1j * self.T * self.H_X)
        self.U_ZZ = linalg.expm(-1j * self.T * self.H_ZZ.reshape([4, 4])).reshape([2, 2, 2, 2])

    def init_up_state(self):
        N = self.Lx * self.Ly
        psi = np.zeros((2,)*N, dtype=complex)
        psi[(0,)*N] = 1.
        self.psi = psi

    def xy_2_idx(self, x, y):
        return x*self.Ly + y

    def idx_2_xy(self, idx):
        return idx//self.Ly, idx%self.Ly

    def init_bond_indices(self):
        """
        construct bond indices for square lattice
        bc: boundary conditions, 1=periodic, 0 =open
        """
        Lx = self.Lx
        Ly = self.Ly
        bc = self.bc
        N = Lx * Ly
        ver_bond_list = []
        hor_bond_list = []
        for idx in range(N):
            x, y = self.idx_2_xy(idx)
            if x < Lx-1 or bc[0]:
                hor_idx = self.xy_2_idx((x+1)%Lx, y)
                hor_bond_list.append([idx, hor_idx])
            if y < Ly-1 or bc[1]:
                ver_idx = self.xy_2_idx(x, (y+1)%Ly)
                ver_bond_list.append([idx, ver_idx])
        bond_list = hor_bond_list + ver_bond_list
        bond_indices = np.array(bond_list)
        self.bonds = bond_indices

    def apply_one_site(self, Op, i):
        self.psi = np.tensordot(Op, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)

    def apply_two_site(self, Op, i, j):
        self.psi = np.tensordot(Op, self.psi, ((2, 3), (i, j)))
        self.psi = np.moveaxis(self.psi, (0, 1), (i, j))

    def apply_U_X(self, i):
        self.psi = np.tensordot(self.U_X, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)

    def apply_U_ZZ(self, i, j):
        self.psi = np.tensordot(self.U_ZZ, self.psi, ((2, 3), (i, j)))
        self.psi = np.moveaxis(self.psi, (0, 1), (i, j))

    def run_Floquet_step(self):
        N = self.Lx * self.Ly
        for i, j in self.bonds:
            self.apply_U_ZZ(i, j)
        for i in range(N):
            self.apply_U_X(i)

    def measure_one_site(self, Op, i):
        Op_psi = np.tensordot(Op, self.psi, (1, i))
        Op_psi = np.moveaxis(Op_psi, 0, i)
        return np.tensordot(self.psi.conj(), Op_psi, self.Lx*self.Ly)

    def measure_one_site_all(self, Op):
        expectation_vals = []
        for i in range(self.Lx * self.Ly):
            expectation_vals.append(self.measure_one_site(Op, i))

        return expectation_vals

    def measure_two_site(self, Op, i, j):
        Op_psi = np.tensordot(Op, self.psi, ((2, 3), (i, j)))
        Op_psi = np.moveaxis(Op_psi, (0, 1), (i, j))
        return np.tensordot(self.psi.conj(), Op_psi, self.Lx*self.Ly)

    def measure_two_site_all_bonds(self, Op):
        expectation_vals = []
        for bond in self.bonds:
            i, j = bond
            expectation_vals.append(self.measure_two_site(Op, i, j))

        return expectation_vals

    def get_energy(self):
        H_X_expectation_vals = self.measure_one_site_all(self.H_X)
        H_ZZ_expectation_vals = self.measure_two_site_all_bonds(self.H_ZZ)
        E = np.sum(H_X_expectation_vals) + np.sum(H_ZZ_expectation_vals)
        assert np.isclose(0, E.imag)
        return E.real



def measure_Z(psi):
    # measure Z expectation value
    N = len(psi.shape)
    z_list = []
    Z = np.matrix(np.array([[1.,0.],[0.,-1.]]))
    for i in range(N):
        psiZ = np.tensordot(Z, psi, (1, i))
        psiZ = np.moveaxis(psiZ, 0, i)
        z_list.append(np.tensordot(psi.conj(), psiZ, N))
    return np.real_if_close(z_list)

def run_ed_floquet(**kwargs):
    print("executing run_simulation() in file", __file__)
    print("got the dictionary kwargs =", kwargs)

    # HERE is where you would usually run your simulation (e.g. DMRG).
    # simulate some heavy calculations:
    Lx = kwargs['Lx']
    Ly = kwargs['Ly']
    T = kwargs['T']
    g = kwargs['g']
    N_steps = kwargs['N_steps']

    model_params = {'Lx': Lx,
                    'Ly': Ly,
                    'bc': [0, 0],
                    'T': T,
                    'g': g,
                    }
    eng = Floquet_Ising_ED(model_params)

    z_time = [measure_Z(eng.psi)]
    E_time = [eng.get_energy()]

    for _ in range(N_steps):
        eng.run_Floquet_step()
        z_time.append(measure_Z(eng.psi))
        E_time.append(eng.get_energy())

    output_filename = kwargs['output_filename']
    print("save results to ", output_filename)
    np.savez(output_filename, z_time=np.array(z_time), E_time=np.array(E_time), **model_params, N_steps=N_steps)

if __name__ == '__main__':
    params = {'Lx': 5,
              'Ly': 5,
              'T': 0.3,
              'g': 4,
              'N_steps': 20,
              'output_filename': 'exact_5_5',
             }
    run_ed_floquet(**params)
