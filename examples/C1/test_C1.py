import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetwork, VisualizeNetwork
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Remesh
    from pyexadis_utils import insert_infinite_line
except ImportError:
    raise ImportError('Cannot import pyexadis')


def init_bcc_Fe_infinite_edge_dislocation(box_length, burg_vec, plane_normal, maxseg=-1):
    """
    Example of a function to generate an initial configuration made of a
    single infinite straight edge dislocation on a BCC 1/2<111>{110} slip
    system, closing on itself through the fully periodic simulation cell.
    """
    print("init_bcc_Fe_infinite_edge_dislocation")
    cell = pyexadis.Cell(h=box_length*np.eye(3), is_periodic=[True,True,True])
    center = np.array(cell.center())

    nodes, segs = [], []
    # theta = 90 deg: line direction perpendicular to the Burgers
    # vector within the glide plane, i.e. an edge dislocation
    nodes, segs = insert_infinite_line(cell, nodes, segs, burg_vec, plane_normal,
                                       origin=center, theta=90.0, maxseg=maxseg)

    N = DisNetManager(ExaDisNet(cell, nodes, segs))
    return N


def test_bcc_Fe_infinite_edge_dislocation():
    """
    Example of a script to perform a simulation of a single infinite
    straight edge dislocation in BCC Fe using the pyexadis binding to
    ExaDiS.
    """
    pyexadis.initialize()

    Lbox = 1000.0
    maxseg = 0.04*Lbox

    # BCC 1/2<111>{110} slip system
    burg_vec = 0.5*np.array([1.0, 1.0, 1.0])
    plane_normal = np.array([-1.0, 1.0, 0.0])

    N = init_bcc_Fe_infinite_edge_dislocation(box_length=Lbox, burg_vec=burg_vec,
                                              plane_normal=plane_normal, maxseg=maxseg)

    vis = VisualizeNetwork()

    # Literature values for BCC Fe
    state = {"burgmag": 2.48e-10, "mu": 85e9, "nu": 0.3, "a": 1.0,
             "maxseg": maxseg, "minseg": 0.01*Lbox, "rann": 2.0}

    calforce  = CalForce(force_mode='LineTension', state=state)
    mobility  = MobilityLaw(mobility_law='SimpleGlide', state=state)
    timeint   = TimeIntegration(integrator='EulerForward', state=state, dt=1.0e-8)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = None
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetwork(calforce=calforce, mobility=mobility, timeint=timeint,
                          collision=collision, topology=topology, remesh=remesh, vis=vis,
                          state=state, max_step=200, loading_mode='stress',
                          applied_stress=np.array([0.0, 0.0, 0.0, 0.0, -4.0e8, 0.0]),
                          print_freq=10, plot_freq=10, plot_pause_seconds=0.0001,
                          write_freq=10, write_dir='output')
    sim.run(N, state)

    pyexadis.finalize()


if __name__ == "__main__":
    test_bcc_Fe_infinite_edge_dislocation()
