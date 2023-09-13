import sys
import numpy as np
import isotns.networks.peps as peps
import isotns.networks.peps_func as peps_func
import isotns.models.tfi_zz_x as tfi_zz_x
from isotns.models.mpo_model import MpoModel, Model, SumMpoModel
import scipy.linalg
import os


Lx = 5
Ly = 5
J = 1.
g = 4.
D = int(sys.argv[1])
D_max = int(sys.argv[2])

dir_name = '%dx%d/TFI_g%.1f_D%d_Dmax%d/' % (Lx, Ly, g, D, D_max)
if not os.path.exists(dir_name):
   os.makedirs(dir_name)

dt = 0.3

Z = np.array([[1., 0.], [0., -1.]])
X = np.array([[0., 1.], [1., 0.]])
ZZ = np.kron(Z, Z)
U_X = scipy.linalg.expm(1j * dt * 4. * X)
U_ZZ = scipy.linalg.expm(1j * dt * ZZ).reshape([2, 2, 2, 2])

tfi_models = tfi_zz_x.TFI(Lx, Ly, J=1, hx=0)
hor_ver_Us = tfi_models.make_U(1.)  # U_ZZ
mpo_model = MpoModel(L=Lx, J=J, g=g, Hamiltonian='TFI')
tfi_models_full = tfi_zz_x.TFI(Lx, Ly, J=1, hx=g)
mpo_model.H_mpo_ver, mpo_model.H_mpo_hor, mpo_model.inv_H_mpo_ver, mpo_model.inv_H_mpo_hor = tfi_models_full.make_ESWN_H_mpo()

trunc_par = {'chi_max': D, 'p_trunc':1e-8}

psi = peps.PEPS(Lx, Ly, [[np.array([1., 0.]).reshape([2,1,1,1,1]) for j in range(Ly)]
                         for i in range(Lx)],
                convention='pldru')

E, E_list = peps_func.compute_energy(psi.copy(), mpo_model, 'mpo', D_max=D_max)
print(E)

E_array = [E,]
Sz_list = [1.]
list_Sz_array = [peps_func.compute_single_site_Op(psi.copy(), Z, D_max)]
list_Sx_array = [peps_func.compute_single_site_Op(psi.copy(), X, D_max)]

cache = peps_func.create_empty_boundary_cache(Lx)
# hor_ver_Us = tfi_models.make_U(3e-1)
for i in range(20):

    # Hhs, Hvs = tfi_models.get_H_()
    Hhs = [[np.tensordot(np.array([[1., 0.], [0., -1.]]), np.eye(2), [[],[]]).transpose([0, 2, 1, 3]) for y in range(Ly)] for x in range(Lx-1)]
    Hvs = [[np.tensordot(np.array([[1., 0.], [0., -1.]]), np.eye(2), [[],[]]).transpose([0, 2, 1, 3]) for y in range(Ly-1)] for x in range(Lx)]

    '''
    # Uhs[x][y] : (Lx-1) x (Ly) terms
    Uhs = [[hor_ver_Us[0][y][x] for y in range(Ly)] for x in range(Lx-1)]
    # Uvs[x][y] : (Lx) x (Ly-1) terms
    Uvs = [[hor_ver_Us[1][x][y] for y in range(Ly-1)] for x in range(Lx)]
    # Uvs_revert[y][x] : (Ly-1) x (Lx) terms
    Uvs_revert = [[hor_ver_Us[1][x][y] for x in range(Lx)] for y in range(Ly-1)]
    '''
    Uhs = [[U_ZZ for y in range(Ly)] for x in range(Lx-1)]
    Uvs = [[U_ZZ for y in range(Ly-1)] for x in range(Lx)]
    Uvs_revert = [[U_ZZ]]

    Hv_, Hh_ = psi.sweep_twocol_VH(Uvs, Uhs, trunc_par, Ovs=Hvs,
                                   Ohs=Hhs,
                                   D_max=D_max, cache=cache)

    for x in range(Lx):
        for y in range(Ly):
            psi[x][y] = np.tensordot(U_X, psi[x][y], [[1], [0]])

    psi.LR_invert()
    psi.UD_invert()
    cache = peps_func.invert_LR_cache(cache)
    cache = peps_func.invert_UD_cache(cache)

    Hv_ = np.squeeze(np.array([Hv_[i]['Os'] for i in range(Lx)]), axis=1).real
    Sz_list.append(Hv_[Lx//2, Ly//2])

    '''
    # The cache cannot be reused when we do sweep_H - rotate - sweep_H -rotate
    Hh_ = psi.sweep_twocol_H(Uhs, trunc_par, D_max=D_max, Os=Hhs)
    psi.rotate()
    Hv_ = psi.sweep_twocol_H(Uvs_revert, trunc_par, D_max=D_max, Os=Hhs)
    # Hv_ = psi.sweep_twocol(hor_ver_Us[0], trunc_par, D_max=D_max, Os=Hhs)
    # psi.rotate_clockwise()
    psi.rotate()
    '''

    # print("Rough energy estimate= ", np.sum(Hh_) + np.sum(Hv_))
    print("Hh_ ", Hh_,
          "Hv_ ", Hv_)
    E, E_list = peps_func.compute_energy(psi.copy(), mpo_model, 'mpo', D_max=D_max)
    psi.print_chi()
    print("True Energy = ", E)
    E_array.append(E)
    list_Sz_array.append(peps_func.compute_single_site_Op(psi.copy(), Z, D_max))
    list_Sx_array.append(peps_func.compute_single_site_Op(psi.copy(), X, D_max))
    psi.save(dir_name + 't%d' % i)

output_filename = 'Lx%d_Ly%d_D%d_D_max%d' % (Lx, Ly, D, D_max)
np.savez(dir_name + output_filename,
         z_time=np.array(list_Sz_array).real,
         x_time=np.array(list_Sx_array).real,
         E_time=np.array(E_array))

# psi.save('Lx%dLy%dD2' % (Lx, Ly))
# np.save('E_list_TEBD_VH.npy', E_array)

