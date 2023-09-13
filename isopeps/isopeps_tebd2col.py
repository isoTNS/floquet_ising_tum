import os
import sys
#--------
#It's important to define nc before importing numpy
# nc = '4'
# os.environ['OMP_NUM_THREADS'] = nc
# os.environ['OPENBLAS_NUM_THREADS'] = nc
# os.environ['MKL_NUM_THREADS'] = nc
# os.environ['VECLIB_MAXIMUM_THREADS'] = nc
# os.environ['NUMEXPR_NUM_THREADS'] = nc
# print('nc    :', nc)

import sys
import random
import numpy as np
import time
from isotns.models.tfi import tfi
from isotns.networks.iso_peps import IsoPEPS
import isotns.networks.peps_func as peps_func
from isotns.misc import *
import scipy.linalg

J       = 1
g       = 4
chi     = int(sys.argv[1])
f       = int(sys.argv[2])
# f       = 8
tchi    = chi * f
Db      = 1
Lx      = 7
Ly      = 8
n       = 0.5
dt      = 0.3
k_iter  = 20
seed    = 0
trunc   = 1e-8
verbose = 0
save    = False
restart = -1 # int(sys.argv[17])
newton  = True
preCon  = False
trunc_first = False
# bool(int(sys.argv[20]))   #Whether truncate first in tebd
m       = 1.  # m = tebd p_trunc/max MMerror

print('newton:', newton)
print('preCon:',  preCon)
#---------------------------------------------------------------
if seed < 0:
    seed = random.randrange(100)
np.random.seed(seed)
fs      = open('info_{}x{}_chi{}_f{}_tchi{}_dt{}_n{}_Db{}_seed{}.out'\
        .format(Lx, Ly, chi, f, tchi, dt, n, Db, seed), 'w+')
fs.write('#count\te\tH_err\tV_err\ttebd_err\n')
#------------------------------------------------------------------------------
if __name__ == '__main__':
    truncation_par = {
        'bond_dimensions': {
            'etaV_max': f*chi,
            'chiV_max': chi,
            'chiH_max': chi,
            'etaH_max': f*chi,
            'tchi_max': tchi,
        },
        't_trunc'  : 1e-8,
        'err_p_tol': 1e-8,
        'err_t_tol': 1e-8,
    }

    measure_truncation_par = {
        'bond_dimensions': {
            'etaV_max': np.amax([4, f])*chi,
            'chiV_max': chi,
            'chiH_max': chi,
            'etaH_max': np.amax([4, f])*chi,
            'tchi_max': np.amax([4, f])*chi,
        },
        't_trunc'  : 1e-8,
        'err_p_tol': 1e-8,
        'err_t_tol': 1e-8,
    }



    model_XX_Z     = tfi(Lx, Ly, g, J)  # -J XX - g Z
    model_XX       = tfi(Lx, Ly, 0, 2)  # -J XX
    from isotns.models.tfi_zz_x import TFI
    model_ZZ       = TFI(Lx, Ly, J=1, hx=0)  # - J ZZ - hx X
    model_ZZ_X     = TFI(Lx, Ly, J=1, hx=4)  # - J ZZ - hx X

    Z = np.array([[1., 0.], [0., -1]])
    X = np.array([[0, 1.], [1., 0]])
    UE, UW = model_XX.make_EW_gates_twocol(dt * 1j)
    UE, UW = model_ZZ.make_EW_gates_twocol(dt * 1j)

    U_Z = scipy.linalg.expm(-1j * dt * -4 * Z)
    U_X = scipy.linalg.expm(-1j * dt * -4 * X)
    U_onsite = U_X
    HE, HW = model_ZZ_X.make_EW_hamiltonian_twocol()
    # HE = np.kron(Z, np.eye(8)).reshape([4, 4, 4, 4])
    # HE = [[HE for i in range(Ly-1)] for k in range(Lx-1)]


    X_bulk = (np.array([1., 1.])/np.sqrt(2)).reshape([2, 1, 1, 1, 1])
    Z_bulk = (np.array([1., 0.])/np.sqrt(1)).reshape([2, 1, 1, 1, 1])
    t_bulk = Z_bulk
    peps    = IsoPEPS(Lx, Ly, list_list_tensor=[[t_bulk]*Ly]*Lx)
    if restart >= 0:
        peps.load('TFI_k{}_{}x{}'.format(restart, Lx, Ly), Lx, Ly)

    es      = []
    dts     = [dt] * k_iter + [0]
    Z_arrays = []
    X_arrays = []
    #------------- Measurement -------------#
    peps_1 = peps.copy()
    Z_data = peps_1.compute_onsite([Z], measure_truncation_par, verbose=True)
    Z_array = np.array(Z_data[1][0])
    Z_arrays.append(Z_array)
    peps_1 = peps.copy()
    X_data = peps_1.compute_onsite([X], measure_truncation_par, verbose=True)
    X_array = np.array(X_data[1][0])
    X_arrays.append(X_array)

    peps_2 = peps.copy()
    info = peps_2.E_tebd_twocol(
            0,
            [None],
            measure_truncation_par,
            Os=[HE],
            N_variational=-1,
            moses=True,
            verbose=verbose,
            splitter_options={
                'n': n,
                'U2_iter': 30,
                'CG_iter': 30,
                'newton_iter': 15,
                'newton': newton,
                'preconditioner': preCon,
                'save_quad': False,
                #'fs_suffix': '',
                },
            sweep_options={
                'trunc_first': trunc_first,
                'm': m,
                'use_last_twocol': False,
                },
            draw=True,
            alt=False,
            )
    e = np.sum(info['expectation_O'])
    es.append(e.real)
    #---------------------------------------#

    for k in range(k_iter):
        print('-' * 40, 'k:', k, '-'*40)
        t0      = time.time()

        info = peps.E_tebd_twocol(
                k,
                [UE],
                truncation_par,
                Os=[HE],
                N_variational=-1,
                moses=True,
                verbose=verbose,
                splitter_options={
                    'n': n,
                    'U2_iter': 30,
                    'CG_iter': 30,
                    'newton_iter': 15,
                    'newton': newton,
                    'preconditioner': preCon,
                    'save_quad': False,
                    #'fs_suffix': '',
                    },
                sweep_options={
                    'trunc_first': trunc_first,
                    'm': m,
                    'use_last_twocol': False,
                    },
                draw=True,
                alt=False,
                )

        e = np.sum(info['expectation_O'])


        err_t   = info['moses_err_t'] #total MM truncation error over the square
        err_p   = info['moses_error'] - info['moses_err_t']
        tebd_err= info['tebd_error']

        for idx_x in range(Lx):
            for idx_y in range(Ly):
                peps[idx_x][idx_y] = np.tensordot(U_onsite, peps[idx_x][idx_y], [[1], [0]])


        peps.LR_invert()

        #------------- Measurement -------------#
        peps_1 = peps.copy()
        Z_data = peps_1.compute_onsite([Z], measure_truncation_par, verbose=True)
        Z_array = np.array(Z_data[1][0])
        Z_arrays.append(Z_array)
        peps_1 = peps.copy()
        X_data = peps_1.compute_onsite([X], measure_truncation_par, verbose=True)
        X_array = np.array(X_data[1][0])
        X_arrays.append(X_array)

        peps_2 = peps.copy()
        info = peps.E_tebd_twocol(
                k,
                [None],
                measure_truncation_par,
                Os=[HE],
                N_variational=-1,
                moses=True,
                verbose=verbose,
                splitter_options={
                    'n': n,
                    'U2_iter': 30,
                    'CG_iter': 30,
                    'newton_iter': 15,
                    'newton': newton,
                    'preconditioner': preCon,
                    'save_quad': False,
                    #'fs_suffix': '',
                    },
                sweep_options={
                    'trunc_first': trunc_first,
                    'm': m,
                    'use_last_twocol': False,
                    },
                draw=True,
                alt=False,
                )
        e = np.sum(info['expectation_O'])
        es.append(e.real)
        #---------------------------------------#
        # When f is large, peps_func is not a good choice.
        # peps_.pwesn_2_pldru()
        # Z_array2 = peps_func.compute_single_site_Op(peps_, O_onsite, D_max=64)


        print('')
        print('dt               :', dts[k])
        print('e                :', np.real(e))
        # print('e_exact          :', e_exact)
        print('check_oc I       :', sum(sum(peps.check_oc(0, Ly-1))))
        print('k{:2} time         :'.format(k), time.time()-t0)

        fs.write('{}\t{}\t{}\t{}\t{}\n'.format(
            k,
            e.real,
            err_p/((Lx-1)*(Ly-1)),
            err_t/((Lx-1)*(Ly-1)),
            tebd_err/((Lx-1)*(Ly-1)),
            ))
        fs.flush()
        if save:
            peps.save('TFI_k{}_{}x{}'.format(k,Lx,Ly))
    fs.close()
    print('E(dt)', es[-5:])
    np.save('Z_arrays_chi%d_f%d_tchi%d.npy' % (chi, f, tchi), Z_arrays)
    np.save('X_arrays_chi%d_f%d_tchi%d.npy' % (chi, f, tchi), X_arrays)

    output_filename = 'Lx%d_Ly%d_chi%d_f%d_tchi%d' % (Lx, Ly, chi, f, tchi)
    np.savez(output_filename,
             z_time=np.array(Z_arrays).real,
             x_time=np.array(X_arrays).real,
             E_time=np.array(es))


'''
    # ----------------------------------------------------- #
    # For debugging
    # ----------------------------------------------------- #
    # ----------------------------------------------------- #
    # 2-3
    # | |
    # 0-1
    I = np.eye(2)
    ZZII = np.kron(np.kron(Z, Z), np.kron(I, I))
    ZIZI = np.kron(np.kron(Z, I), np.kron(Z, I))
    IIZZ = np.kron(np.kron(I, I), np.kron(Z, Z))
    IZIZ = np.kron(np.kron(I, I), np.kron(Z, Z))
    U_ZZII = scipy.linalg.expm(1j * dt * ZZII)
    U_ZIZI = scipy.linalg.expm(1j * dt * ZIZI)
    U_IIZZ = scipy.linalg.expm(1j * dt * IIZZ)
    U_IZIZ = scipy.linalg.expm(1j * dt * IZIZ)

    U_base = U_ZZII.dot(U_ZIZI)
    U_up = U_base.dot(U_IIZZ)
    U_right = U_base.dot(U_IZIZ)
    U_end = U_up.dot(U_IZIZ)

    U_base = U_base.reshape([4, 4, 4, 4])
    U_up = U_up.reshape([4, 4, 4, 4])
    U_right = U_right.reshape([4, 4, 4, 4])
    U_end = U_end.reshape([4, 4, 4, 4])
    UE_new = [[U_base] * (Ly-2) + [U_up]] * (Lx - 2)
    UE_new = UE_new + [[U_right] * (Ly-2) + [U_end]]
    # ----------------------------------------------------- #
'''
