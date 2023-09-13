import matplotlib.pyplot as plt
import numpy as np
import plot_setup
plot_setup.setup(10)


exact_data = np.load('exact/exact_5_5.npz')

PEPS_data = {}
# for D in [2, 4, 6, 8]:
for D in [2, 3, 4, 5, 6]:
    PEPS_data[D] = np.load('peps/5x5/TFI_g4.0_D%d_Dmax64/Lx5_Ly5_D%d_D_max64.npz' % (D, D))

iso_PEPS_data = {}
# for chi, f, tchi in [(4, 4, 16), (8, 4, 32), (8, 8, 64)]:
f = 2
for chi, f, tchi in [(2, f, 2*f), (4, f, 4*f), (8, f, 8*f), (12, f, 12*f)]:
    try:
        iso_PEPS_data[(chi, f, tchi)] = np.load('isopeps/Lx5_Ly5_chi%d_f%d_tchi%d.npz' % (chi, f, tchi))
    except:
        pass


save = False
filetype = 'png'

'''
chi = 4
f = 4
tchi = 16
A = np.load('isopeps/O_arrays_chi%d_f%d_tchi%d.npy' % (chi, f, tchi))



A = A.reshape([-1, 25])
A_ = np.ones([A.shape[0] + 1, 25])
A_[1:, :] = A[:, :]
A = A_
'''

# plt.figure()
# plt.imshow(A, origin='lower')
# plt.colorbar()
# if save:
#     plt.savefig('TEBD2_chi%d_f%d_tchi%d.png' % (chi, f, tchi))
#
# plt.figure()
# plt.imshow(exact_data['z_time'].reshape([-1, 25]), origin='lower')
# plt.colorbar()
# if save:
#     plt.savefig('exact.png')



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
fig, axes = plt.subplots(2, 1, figsize=(3.5,4.3), sharex=True)
exact_mean = np.mean(exact_data['z_time'].reshape([-1, 25]), axis=-1)
axes[0].plot(np.mean(exact_data['z_time'].reshape([-1, 25]), axis=-1), 'ko-', label='exact')
for key in iso_PEPS_data.keys():
    axes[0].plot(np.mean(iso_PEPS_data[key]['z_time'].reshape([-1, 25]), axis=-1), 's--', markerfacecolor='none', label='D=%d f=%d tchi=%d' % key)
    axes[1].plot(np.abs(np.mean(iso_PEPS_data[key]['z_time'].reshape([-1, 25]), axis=-1) - exact_mean),
                 's--', markerfacecolor='none', label='D=%d f=%d tchi=%d' % key)

axes[1].legend()
axes[0].set_ylabel(u'$\\langle \\bar{S_z} \\rangle$')
axes[1].set_ylabel(u'$\\epsilon$')
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95, hspace=0.03)

plt.savefig('figures/avg_Z_isoPEPS_f%d.png' % (f), transparent=True)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
fig, axes = plt.subplots(2, 1, figsize=(3.5,4.3), sharex=True)
axes[0].plot(exact_data['E_time'], 'ko-', label='exact')
for key in iso_PEPS_data.keys():
    axes[0].plot(iso_PEPS_data[key]['E_time'], 's--', markerfacecolor='none', label='D=%d f=%d tchi=%d' % key)
    axes[1].plot(np.abs(iso_PEPS_data[key]['E_time'] - exact_data['E_time']), 's--', markerfacecolor='none',
                 label='D=%d f=%d tchi=%d' % key)

axes[1].legend()
axes[0].set_ylabel(u'$E$')
axes[1].set_ylabel(u'$\\epsilon_E$')
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95, hspace=0.03)

plt.savefig('figures/E_isoPEPS_f%d.png' % f, transparent=True)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
fig, axes = plt.subplots(2, 1, figsize=(3.5,4.3), sharex=True)
exact_mean = np.mean(exact_data['z_time'].reshape([-1, 25]), axis=-1)
axes[0].plot(np.mean(exact_data['z_time'].reshape([-1, 25]), axis=-1), 'ko-', label='exact')
# axes[0].plot(np.mean(A.reshape([-1, 25]), axis=-1), 'o--', markerfacecolor='none', label='TEBD2')
# axes[1].plot(np.abs(np.mean(A.reshape([-1, 25]), axis=-1) - exact_mean),
#              'o--', markerfacecolor='none', label='TEBD2')
for key in PEPS_data.keys():
    axes[0].plot(np.mean(PEPS_data[key]['z_time'].reshape([-1, 25]), axis=-1), 's--', markerfacecolor='none', label='PEPS D=%d' % key)
    axes[1].plot(np.abs(np.mean(PEPS_data[key]['z_time'].reshape([-1, 25]), axis=-1) - exact_mean),
                 's--', markerfacecolor='none', label='PEPS D=%d' % key)

axes[1].legend()
axes[0].set_ylabel(u'$\\langle \\bar{S_z} \\rangle$')
axes[1].set_ylabel(u'$\\epsilon$')
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95, hspace=0.03)

plt.savefig('figures/avg_Z_PEPS.png', transparent=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

fig, axes = plt.subplots(2, 1, figsize=(3.5,4.3), sharex=True)
axes[0].plot(exact_data['E_time'], 'ko-', label='exact')
for key in PEPS_data.keys():
    axes[0].plot(PEPS_data[key]['E_time'], 's--', markerfacecolor='none', label='PEPS D=%d' % key)
    axes[1].plot(np.abs(PEPS_data[key]['E_time'] - exact_data['E_time']), 's--', markerfacecolor='none',
                 label='PEPS D=%d' % key)

axes[1].legend()
axes[0].set_ylabel(u'$E$')
axes[1].set_ylabel(u'$\\epsilon_E$')
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95, hspace=0.03)

plt.savefig('figures/E_PEPS.png', transparent=True)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



plt.show()
