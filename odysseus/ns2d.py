"""
Module for direct numerical simulations of the 2D Navier-Stokes equations using Dedalus.

For instance, from ipython:
> from odysseus import ns2d
> ns2d.NavierStokesDecaying2D(128, 128, nu=5e-3, alpha=1e-2).run_simul(fout=None, dt=5e-3, initial='random', kf=10)
> ns2d.NavierStokesForced2D(128, 128, kf=10, alpha=1e-2, nu=5e-3, ampl=1000).run_simul(fout=None, dt=5e-3, initial='rest')

Or directly from the shell:

`python -c "from odysseus import ns2d; ns2d.NavierStokesDecaying2D(128, 128, nu=5e-3, alpha=1e-2).run_simul(fout=None, dt=5e-3, initial='random', kf=10)"`

"""
import os
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
from mpi4py import MPI
from dedalus import public as de
from .base import DedalusSimulation, Monitor

class MonitorNS2D(Monitor):
    """
    Monitor class for the Navier-Stokes equations in 2D.
    """
    def __init__(self, domain, solver):
        Monitor.__init__(self, domain, solver)
        self.ax_xy, self.ax_spec, self.ax_time, self.ax_time2 = self.create_axes(solver.stop_sim_time)

        x = domain.grid(0)
        y = domain.grid(1)
        self.xx, self.yy = np.meshgrid(x, y)

        omega = solver.state['omega']
        self.quad, self.cax, self.line_spec, self.line_kin, self.line_ens = self.setup_monitor(omega)
        self.simul_energy = []
        self.simul_enstrophy = []
        self.simul_time = []

    def create_axes(self, simtime):
        ax_xy = self.fig.add_subplot(2, 2, 1)
        ax_xy.set_aspect('equal')
        ax_xy.set_xlabel(r'$x$')
        ax_xy.set_ylabel(r'$y$')
        ax_xy.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2, np.pi))
        ax_xy.set_xticklabels((r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
        ax_xy.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2, np.pi))
        ax_xy.set_yticklabels((r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))

        ax_spec = self.fig.add_subplot(2, 2, 2)
        ax_spec.set_xlabel(r'$k$')
        ax_spec.set_ylabel(r'$Ek$')
        ax_spec.set_xscale('log')
        ax_spec.set_yscale('log')

        spec = mpl.gridspec.GridSpec(ncols=1, nrows=2)
        ax_time = self.fig.add_subplot(spec[1])
        ax_time.set_xlabel(r'$t$')
        ax_time.set_ylabel(r'$E$')
        ax_time.set_xlim((0, simtime))
        ax_time.set_ylim((0, 1))
        ax_time2 = ax_time.twinx()
        ax_time2.set_ylabel(r'$Z$')

        return ax_xy, ax_spec, ax_time, ax_time2

    def setup_monitor(self, omega):
        norm = mpl.colors.Normalize(vmin=np.min(omega['g']), vmax=np.max(omega['g']))
        quad = self.ax_xy.contourf(self.xx, self.yy, omega['g'].T,
                                   cmap='RdBu_r', norm=norm, extend='both')
        ax_divider = make_axes_locatable(self.ax_xy)
        cax = ax_divider.append_axes("right", size="2%", pad="1%")
        cbar = plt.colorbar(quad, cax=cax, label=r'$\omega$')

        line_spec, = self.ax_spec.plot([0], [0], color='C0')

        line_kin, = self.ax_time.plot([0], [0], color='C0')
        line_ens, = self.ax_time2.plot([0], [0], color='C1')
        plt.tight_layout()
        plt.pause(0.1)
        return quad, cax, line_spec, line_kin, line_ens

    def update_monitor(self, domain, solver):
        x = domain.grid(0)
        y = domain.grid(1)
        omega = solver.state['omega']
        psi = solver.state['psi']

        for coll in self.quad.collections:
            coll.remove()
        norm = mpl.colors.Normalize(vmin=np.min(omega['g']), vmax=np.max(omega['g']))
        self.quad = self.ax_xy.contourf(self.xx, self.yy, omega['g'].T,
                                        cmap='RdBu_r', norm=norm, extend='both')

        pts = self.cax.get_position().get_points()
        label = self.cax.get_ylabel()
        self.cax.remove()
        self.cax = self.fig.add_axes([pts[0][0], pts[0][1],
                                      pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]])
        cbar = plt.colorbar(self.quad, cax=self.cax)
        cbar.ax.set_ylabel(label)

        self.line_spec.remove()
        spectrum = energy_spectrum(128, omega['c'], psi['c'])
        self.line_spec, = self.ax_spec.plot(spectrum, color='C0')
        self.ax_spec.set_ylim((0.8*np.min(spectrum), 1.2*np.max(spectrum)))

        self.simul_time += [solver.sim_time]
        self.simul_energy += [energy(omega['g'], psi['g'], x, y)]
        self.simul_enstrophy += [enstrophy(omega['g'], x, y)]
        self.line_kin.remove()
        self.line_kin, = self.ax_time.plot(np.array(self.simul_time), np.array(self.simul_energy),
                                           color='C0')
        self.ax_time.set_ylim((0, 1.5*np.max(self.simul_energy)))
        self.line_ens.remove()
        self.line_ens, = self.ax_time2.plot(np.array(self.simul_time),
                                            np.array(self.simul_enstrophy), color='C1')
        self.ax_time2.set_ylim((0, 1.2*np.max(self.simul_enstrophy)))
        plt.pause(0.001)



class NavierStokesDecaying2D(DedalusSimulation):
    """
    Class for simulating the freely decaying Navier-Stokes equations using Dedalus on a 2D domain
    periodic in both directions.
    """
    # default parameter values:
    params = {'Lx': 2*np.pi, 'Ly': 2*np.pi, 'alpha': 1e-1, 'nu': 0.1}
    monitor_class = MonitorNS2D

    def __init__(self, nx, ny, **kwargs):
        r"""
        Parameters
        ----------
        nx, ny: integers
        The resolution in :math:`x` and :math:`y` (periodic BC, Fourier).

        Keyword Arguments
        -----------------
        Lx, Ly: float
        The domain size. By default Lx and Ly are both 2\pi.
        """
        #update parameters:
        self.params.update(kwargs)
        x_basis = de.Fourier('x', nx, interval=(-self.params['Lx']/2, self.params['Lx']/2))
        y_basis = de.Fourier('y', ny, interval=(-self.params['Ly']/2, self.params['Ly']/2))
        DedalusSimulation.__init__(self, de.Domain([x_basis, y_basis], grid_dtype=np.float64))

    def __str__(self):
        return 'Navier-Stokes equations on a 2D square domain with periodic boundary conditions.'

    def setup_ivp(self):
        r"""
        Build and return the dedalus problem representation for the 2D Navier-Stokes equations.

        Notes
        -----
        This solve the equations:
        :math:`\partial_t \omega + \{ \psi, \omega \} = \nu \Delta \omega -\alpha \omega`

        with :math:`\omega=-\Delta \psi` the vorticity, :math:`\psi` the stream function.
        """
        # setup the Initial Value Problem with the Navier-Stokes Equations
        problem = de.IVP(self.domain, variables=['omega', 'psi'])

        problem.parameters.update(self.params.items())

        problem.substitutions['ke'] = '0.5*omega*psi/(Lx*Ly)'

        problem.add_equation('dt(omega) + alpha*omega -nu*(dx(dx(omega))+dy(dy(omega))) = dx(psi)*dy(omega)-dx(omega)*dy(psi)')
        problem.add_equation('omega+dx(dx(psi))+dy(dy(psi)) = 0', condition="(nx != 0) or (ny != 0)")
        problem.add_equation("psi = 0", condition="(nx == 0) and (ny == 0)")

        return problem

    def setup_initial_conditions(self, solver, **kwargs):
        x = self.domain.grid(0)
        y = self.domain.grid(1)
        omega = solver.state['omega']
        psi = solver.state['psi']

        initial_condition = kwargs.get('initial', 'kolmogorov')
        if initial_condition == 'kolmogorov':
            omega['g'], psi['g'] = kolmogorov_flow(x, y, kwargs.get('k', 2), kwargs.get('l', 2))
        elif initial_condition == 'random':
            nx, ny = self.domain.global_grid_shape(scales=1)
            if nx != ny:
                raise ValueError('Random initial condition available only on a square domain')
            psi['g'] = generate_forcing2d(psi['g'].shape, kwargs.get('kf', 20))
            omega['g'] = vorticity(psi)['g']
            norm = np.sqrt(energy(omega['g'], psi['g'], x, y))
            psi['g'] = psi['g']/norm
            omega['g'] = omega['g']/norm
        elif initial_condition == 'rest':
            psi['g'] = np.zeros(psi['g'].shape)
            omega['g'] = np.zeros(omega['g'].shape)
        else:
            raise ValueError('Initial condition unknown')

    @classmethod
    def setup_output_data(cls, fout, solver, **kwargs):
        """
        Define the analysis tasks: output full fields or other diagnostics.
        """
        bin_output_freq = {k: kwargs[f"binary_{k}"] for k in ('sim_dt', 'wall_dt', 'iter')
                           if f"binary_{k}" in kwargs}
        glb_output_freq = {k: kwargs[f"global_{k}"] for k in ('sim_dt', 'wall_dt', 'iter')
                           if f"global_{k}" in kwargs}
        analysis = solver.evaluator.add_file_handler(f"{fout}_bin", **bin_output_freq)
        analysis.add_task('omega')
        analysis = solver.evaluator.add_file_handler(f"{fout}_global", **glb_output_freq)
        analysis.add_task("integ(integ(ke, 'x'), 'y')", name='E')
        analysis.add_task("integ(integ(omega**2, 'x'), 'y')", name='Z')

    @classmethod
    def post_processing(cls, fout):
        if not os.path.exists(fout):
            os.makedirs(fout)
        for out_kind in ('bin', 'global'):
            DedalusSimulation.post_processing(f"{fout}_{out_kind}")
            os.rename(f"{fout}_{out_kind}.h5", os.path.join(fout, f"{fout}_{out_kind}.h5"))
        cls.logger.info(f"Output files moved to: {fout}")

    @classmethod
    def auto_output_file_name(cls, **kwargs):
        alpha = kwargs['alpha']
        nu = kwargs['nu']
        filename = f"ns2d_decaying_a{alpha}_nu{nu}"
        return filename.replace('.', '') #dedalus does not accept dots in output folder names


def wavevectors2d(nn):
    """
    Generate the wave vectors ka and ka2 in 2D and return them.
    ka  is defined by ka(i)=i for 0 <= i <= nn/2-1 and ka(i)=i-nn for nn/2 <= i <= nn-1
    ka2 is defined by ka2(i, j)=ka(i)**2+ka(j)**2 for 0 <= i, j <= nn-1
    These wave vectors are useful for all the pseudospectral operations.
    """
    ka = np.concatenate((np.arange(int(nn/2), dtype=np.int32),
                         np.arange(-int(nn/2), 0, dtype=np.int32)), axis=0)
    ka2 = np.tile(ka**2, (nn, 1))
    ka2 = ka2+ka2.T
    return ka, ka2

def generate_forcing2d_fourier(shape, kf, seed=None):
    """
    Generate a 2D isotropic and homogeneous random field with prescribed correlation length.
    Return the result in Fourier space.

    Efficient Python implementation with boolean array indexing.
    size=???
    """
    nx = shape[0]
    ny = int(shape[1]/2+1)
    fk = np.zeros((nx, ny), dtype=np.complex64)
    _, ka2 = wavevectors2d(nx)
    np.random.seed(seed)
    sel = (kf**2 <= ka2[:, :ny])*(ka2[:, :ny] <= (kf+1)**2)
    indk, indl = sel.nonzero()
    for k in indk[(indl == 0)*(indk < nx/2)]:
        fk[k, 0] = np.exp(2*np.pi*1j*np.random.random())
        fk[nx-k, 0] = np.conj(fk[k, 0])
    for k, l in zip(indk[indl > 0], indl[indl > 0]):
        fk[k, l] = np.exp(2*np.pi*1j*np.random.random())
    return fk

def generate_forcing2d(shape, kf, method=generate_forcing2d_fourier, seed=None):
    """
    Generate a 2D isotropic and homogeneous random field with prescribed correlation length.
    Return the result in real space.
    """
    return np.fft.irfftn(method(shape, kf, seed=seed), norm='ortho')

def energy(omega, psi, x, y):
    """
    Compute and return the energy per unit area.

    Parameters
    ----------
    omega, psi, x, y: 2D numpy arrays
        x and y should be the dedalus grid: x=domain.grid(0), y=domain.grid(1)

    Returns
    -------
    energy (float): The kinetic energy
    """
    area = np.trapz(np.trapz(np.ones_like(omega), x=y), x=x[:, 0])
    return np.trapz(np.trapz(0.5*omega*psi, x=y), x=x[:, 0])/area

def enstrophy(omega, x, y):
    """
    Compute and return the enstrophy per unit area.

    Parameters
    ----------
    omega, x, y: 2D numpy arrays
        x and y should be the dedalus grid: x=domain.grid(0), y=domain.grid(1)

    Returns
    -------
    enstrophy (float): The enstrophy
    """
    area = np.trapz(np.trapz(np.ones_like(omega), x=y), x=x[:, 0])
    return np.trapz(np.trapz(0.5*omega**2, x=y), x=x[:, 0])/area

def energy_spectrum(nn, omegak, psik):
    """
    Compute and return the spectrum of energy per unit area.

    Parameters
    ----------
    omegak, psik: 2D numpy arrays containing the Fourier coefficients of the fields.
        omegak = omega['c'], psik = psi['c']

    Notes
    -----
    This code is directly adapted from GHOST.
    It is probably not the most efficient (or clear) python implementation.
    """
    Ek = np.zeros(int(nn/2)+1)
    _, ka2 = wavevectors2d(int(nn))
    for j in range(nn-1):
        kmn = int(np.sqrt(ka2[j, 1])+.5)
        if (kmn > 0) and (kmn <= int(nn/2)+1):
            Ek[kmn-1] += 0.5*np.abs(omegak[0, j]*psik[0, j])
    for i in range(1, int(nn/2)):
        for j in range(nn-1):
            kmn = int(np.sqrt(ka2[j, i])+.5)
            if (kmn > 0) and (kmn <= int(nn/2)+1):
                Ek[kmn-1] += np.abs(omegak[i, j]*psik[i, j])
    return Ek


def kolmogorov_flow(x, y, k, l):
    """
    Vorticity and stream function for a Kolmogorov flow.

    Parameters
    ----------
    x, y: 2D numpy arrays
          x and y should be the dedalus grid: x=domain.grid(0), y=domain.grid(1)
    k, l: int, the wave number in the x and y directions.

    Returns
    -------
    omega, psi: 2D numpy arrays, vorticity field and stream function.

    Notes
    -----
    The fields are normalized so that the energy is always one.
    """
    psi = np.cos(k*x)*np.cos(l*y)
    omega = (k**2+l**2)*psi
    norm = np.sqrt(energy(omega, psi, x, y))
    return omega/norm, psi/norm

def vorticity(psi):
    """
    Compute vorticity from streamfunction using Dedalus.

    Parameters
    ----------
    psi: numpy array (2D)
        The stream function.

    Notes
    -----
    This is just the opposite of the Laplacian operator.
    """
    psi_xx = psi.differentiate(x=2)
    psi_yy = psi.differentiate(y=2)
    omega_op = -psi_xx-psi_yy
    return omega_op.evaluate()

class NavierStokesForced2D(NavierStokesDecaying2D):
    """
    Class for simulating the freely decaying Navier-Stokes equations using Dedalus on a 2D domain
    periodic in both directions.
    """
    # default parameter values:
    params = {'Lx': 2*np.pi, 'Ly': 2*np.pi, 'alpha': 1e-1, 'nu': 0.1, 'kf': 10, 'ampl': 0.1}

    def __str__(self):
        return 'Forced Navier-Stokes equations on a 2D square domain with periodic boundary conditions.'

    def setup_ivp(self):
        r"""
        Build and return the dedalus problem representation for the forced 2D Navier-Stokes
        equations.

        Notes
        -----
        This solve the equations:
        :math:`\partial_t \omega + \{ \psi, \omega \} = \nu \Delta \omega -\alpha \omega+f`

        with :math:`\omega=-\Delta \psi` the vorticity, :math:`\psi` the stream function.
        """
        shape = self.domain.local_grid_shape(scales=1)

        def Forcing(*args):
            kf = args[0].value
            ampl = args[1].value
            return ampl*generate_forcing2d(shape, kf)

        def F(*args, domain=self.domain, g=Forcing):
            return de.operators.GeneralFunction(self.domain, layout='g', func=g, args=args)

        de.operators.parseables['F'] = F

        # setup the Initial Value Problem with the Navier-Stokes Equations
        problem = de.IVP(self.domain, variables=['omega', 'psi'])

        problem.parameters.update({k: v for k, v in self.params.items() if k in ('alpha', 'nu', 'kf', 'ampl')})

        problem.add_equation('dt(omega) + alpha*omega -nu*(dx(dx(omega))+dy(dy(omega))) = dx(psi)*dy(omega)-dx(omega)*dy(psi)+ F(kf, ampl)')
        problem.add_equation('omega+dx(dx(psi))+dy(dy(psi)) = 0', condition="(nx != 0) or (ny != 0)")
        problem.add_equation("psi = 0", condition="(nx == 0) and (ny == 0)")

        return problem

    @classmethod
    def auto_output_file_name(cls, **kwargs):
        alpha = kwargs['alpha']
        nu = kwargs['nu']
        filename = f"ns2d_forced_a{alpha}_nu{nu}"
        return filename.replace('.', '') #dedalus does not accept dots in output folder names

if __name__ == '__main__':
    NavierStokesDecaying2D(128, 128, nu=5e-3, alpha=1e-2).run_simul(fout=None, dt=5e-3, initial='random', kf=10)
    #NavierStokesForced2D(128, 128, initial='rest', kf=10, alpha=1e-2, nu=5e-3, ampl=1000).run_simul(fout=None, dt=5e-3)
