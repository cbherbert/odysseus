"""
Base class for fluid simulations with Dedalus.
"""
import time
import logging
import pathlib
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de
from dedalus.tools import post

root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

class Monitor:
    """
    Abstract monitor class.

    Monitors can be used to display live information about the run. For instance, it can be a
    contour plot of a 2D field, a velocity profile, an energy spectrum, or a plot of global
    quantities (like the energy) as a function of time, or a combination of the above.

    This class should be subclassed to decide which quantities should be plotted for your
    simulation.
    """
    def __init__(self, domain, solver):
        self.fig = plt.figure()

    def update_monitor(self, domain, solver):
        raise NotImplementedError("update_monitor should be implemented by the simulation subclass!")

class DedalusSimulation:
    """
    Abstract class for simulating an IVP with Dedalus.

    This class essentially contains a general loop for running simulations.
    It can be customized by overriding the methods called within the loops in a subclass.

    Subclasses should implement the methods `setup_ivp` and `setup_initial_conditions`.
    The method `auto_output_file_name` is optional.
    """
    monitor_class = Monitor
    logger = logging.getLogger(__name__)

    def __init__(self, domain):
        self.domain = domain

    def setup_ivp(self):
        """
        Create the IVP problem object and return it.
        """
        raise NotImplementedError('setup_ivp should be implemented by the simulation subclass!')

    def setup_initial_conditions(self, solver, **kwargs):
        """
        Set the initial state of the solver object given as an argument.
        """
        raise NotImplementedError('setup_initial_conditions should be implemented by the simulation subclass!')

    @classmethod
    def auto_output_file_name(cls, **kwargs):
        """
        Construct and return a filename based on the simulation subclass and the parameters of the
        problem.
        """
        raise NotImplementedError('auto_output_file_name should be implemented by the simulation subclass!')

    @classmethod
    def setup_output_data(cls, fout, solver, **kwargs):
        """
        Define the analysis tasks: output full fields or other diagnostics.
        """
        raise NotImplementedError('setup_output_data should be implemented by the simulation subclass!')

    @classmethod
    def post_processing(cls, fout):
        """
        Carry out post-processing tasks, like merging output files.
        """
        post.merge_process_files(fout, cleanup=True)
        set_paths = list(pathlib.Path(fout).glob(f"{fout}_s*.h5"))
        post.merge_sets(f"{fout}.h5", set_paths, cleanup=True)
        shutil.rmtree(fout)

    @staticmethod
    def getscheme(scheme):
        """
        This is a convenience method to choose the timestepping scheme.
        It is defined for use with older versions of dedalus which did not allow to select the
        scheme by label, to avoid having to import dedalus in scripts to pass the scheme to
        odysseus. It should not be necessary with more recent versions of dedalus.
        """
        dic = {'CNAB1': de.timesteppers.CNAB1,
               'SBDF1': de.timesteppers.SBDF1,
               'CNAB2': de.timesteppers.CNAB2,
               'MCNAB2': de.timesteppers.MCNAB2,
               'SBDF2': de.timesteppers.SBDF2,
               'CNLF2': de.timesteppers.CNLF2,
               'SBDF3': de.timesteppers.SBDF3,
               'SBDF4': de.timesteppers.SBDF4,
               'RK111': de.timesteppers.RK111,
               'RK222': de.timesteppers.RK222,
               'RK443': de.timesteppers.RK443,
               'RKSMR': de.timesteppers.RKSMR}
        return dic.get(scheme, dic['RK443'])

    def run_simul(self, **kwargs):
        """
        General loop for running a simulation.

        Keyword Arguments
        -----------------
        monitor: bool, display monitor during run integration or not (default is True)
        dt: float, the integration time step (no adaptive time-step yet)
        fout: str, the output file name (default is None)
              'auto' builds file name from run parameters, for classes which implement it.

        sim_time: float, total simulation time (default 20)
        wall_time: float, wall time limit (default is infinite)
        max_iter: int, maximum number of iterations (default is 10 000 000)
        log_iter: int, frequency for log info (default is 10)

        Typically you will only use one of the three below:
        sim_dt: float, output frequency using simulatin time (default is 0.1)
        wall_dt: float, output frequency using wall time
        iter: int, output frequency using iteration number
        """
        do_monitor = kwargs.pop('monitor', True)
        dt = kwargs.pop('dt', 0.05)
        fout = kwargs.pop('fout', None)

        problem = self.setup_ivp()

        # Setup the time stepping scheme and build the solver:
        ts = self.getscheme(kwargs.get('timestepper', 'RK443'))
        solver = problem.build_solver(ts)

        self.setup_initial_conditions(solver, **kwargs)

        solver.stop_sim_time = kwargs.pop('sim_time', 20.)
        solver.stop_wall_time = kwargs.pop('wall_time', np.inf)
        solver.stop_iteration = kwargs.pop('max_iter', 10000000)
        log_iter = kwargs.pop('log_iter', 10)

        if fout == 'auto':
            fout = self.auto_output_file_name(**problem.parameters)

        if fout is not None:
            self.setup_output_data(fout, solver, **kwargs)

        if do_monitor:
            monitor = self.monitor_class(self.domain, solver)

        try:
            self.logger.info('Starting loop')
            start_time = time.time()
            run_time = start_time
            while solver.ok:
                dt = solver.step(dt, trim=False)
                if (solver.iteration-1) % log_iter == 0:
                    iter_time = (time.time()-run_time)/log_iter
                    run_time = time.time()
                    self.logger.info(f"Iteration: {solver.iteration:03}, Time: {solver.sim_time:e}, dt: {dt:e}, time per step: {iter_time:e} s")
                    if do_monitor:
                        monitor.update_monitor(self.domain, solver)
        except:
            self.logger.error('Exception raised, triggering end of main loop.')
            raise
        finally:
            end_time = time.time()
            self.logger.info(f"Iterations: {solver.iteration}")
            self.logger.info(f"Sim end time: {solver.sim_time:.2f}")
            self.logger.info(f"Run time: {end_time-start_time:.2f} sec")
            self.logger.info(f"Run time: {(end_time-start_time)/3600*self.domain.dist.comm_cart.size:.3f} cpu-hr")
            if fout is not None:
                self.post_processing(fout)

            if do_monitor:
                plt.show()
            else:
                try:
                    subprocess.run(['notify-send', '-i', pathlib.Path('dedalus.png').resolve(),
                                    '-u', 'critical', 'Dedalus',
                                    f"Simulation finished\nOutput File:{fout}.h5"])
                except:
                    pass
