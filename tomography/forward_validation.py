
import numpy as np

import salvus.namespace as sn

import salvus.mesh
import salvus.mesh.layered_meshing as lm
from my_code.utilities import *
from pathlib import Path
import salvus.project as sp

from material_ela_constants.Elastic_Material import Austenite
from salvus.material import elastic, orientation
from salvus.material._details.material import to_solver_form
from salvus.material._details import material as md

from datetime import datetime

# Directories in WSL
PROJECT_DIR = '/home/oliver/workspace/Salvus/elastic_model/anisotropic/Project'
IMAGE_DIR = '/home/oliver/workspace/Salvus/elastic_model/anisotropic/image'
DATA_DIR = '/home/oliver/workspace/Salvus/elastic_model/anisotropic/data'


# Directories in Windows
PROJECT_DIR_WIN = '/mnt/d/Salvus_project/elastic_model/anisotropic/Project'
DATA_DIR_WIN = '/mnt/d/Salvus_project/elastic_model/anisotropic/data'
IMAGE_DIR_WIN = '/mnt/d/Salvus_project/elastic_model/anisotropic/image'


# create dir if it does not exist
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(IMAGE_DIR_WIN).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR_WIN).mkdir(parents=True, exist_ok=True)




# Salvus site name
SALVUS_FLOW_SITE_NAME = 'oliver_wsl'
RANKS_PER_JOB = 8

# 1 MHz should run on laptops, 4 MHz and higher we recommend GPUs
CENTRAL_FREQUENCY = 1e6  # MHz

assert CENTRAL_FREQUENCY >= 1e6

PROJECT_NAME = f"forward_validation_box"




x0, x1 = 0.0, 0.01
y0, y1 = 0.0, 0.01

domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)
p = sn.Project.from_domain(
    path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True
)





# Event for simulation
bdr_gap = 0 
srcs_pos = Vector(x1/8, y1 * (1-bdr_gap))
dir = (0, 1)
srcs = VectorPoint2D(x=srcs_pos.x,y=srcs_pos.y, fx=dir[0], fy=dir[1]) 

n_rxs = 9
np.linspace(0.1*x1,0.9*x1,n_rxs)

rxs_pos = [Vector(x, y1 * bdr_gap) for x in np.linspace(0.1*x1,0.9*x1,n_rxs)]


events = []
fileds = ["displacement"]     # received fileds

rxs = [Point2D(x=r.x, y=r.y,
        station_code=f"REC_{i + 1}",
        # Note that one can specify the desired recording field here.
        fields=fileds,)
    for i, r in enumerate(rxs_pos)
    ]


events.append(sn.Event(event_name=f"event_0", sources=srcs, receivers=rxs))


# add the events to Project
add_events_to_Project(p, events)






# Mesh for simulation
homogeneous_model = lm.LayeredModel(
    [
        sn.material.elastic.Velocity.from_params(rho=2600.0, vp=5000.0, vs=2500),
    ]
)

homogeneous_model_ab = sn.layered_meshing.MeshingProtocol(
    homogeneous_model,
        # ab=salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
        #     free_surface=False,
        #     number_of_wavelengths=1.5,
        #     reference_velocity=5000,
        #     reference_frequency=CENTRAL_FREQUENCY * 2,),
)
    




ele_per_wavelength_ls = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
model_order = 4

for element_per_wavelength in ele_per_wavelength_ls:
    mesh_homogeneous = lm.mesh_from_domain(
        domain=domain,
        model=homogeneous_model_ab,
        mesh_resolution=sn.MeshResolution(
            reference_frequency=CENTRAL_FREQUENCY * 2, 
            elements_per_wavelength=element_per_wavelength,
            model_order= model_order)
    )

    mesh_heterogeneous = mesh_homogeneous.copy()
    centroids = mesh_heterogeneous.get_element_centroid()

    mask_x_1 = (centroids[:,0] >= x1 * 0/4) & (centroids[:,0] <= x1 * 1/4) 
    mask_x_2 = (centroids[:,0] > x1 * 1/4) & (centroids[:,0] <= x1 * 2/4)
    mask_x_3 = (centroids[:,0] > x1 * 2/4) & (centroids[:,0] <= x1 * 3/4)
    mask_x_4 = (centroids[:,0] > x1 * 3/4) & (centroids[:,0] <= x1 * 4/4)
    mask_y_1 = (centroids[:,1] >= y1 * 0/4) & (centroids[:,1] <= y1 * 1/4)
    mask_y_2 = (centroids[:,1] > y1 * 1/4) & (centroids[:,1] <= y1 * 2/4)
    mask_y_3 = (centroids[:,1] > y1 * 2/4) & (centroids[:,1] <= y1 * 3/4)
    mask_y_4 = (centroids[:,1] > y1 * 3/4) & (centroids[:,1] <= y1 * 4/4)

    defect_1 = (mask_x_1 | mask_x_3) & (mask_y_1 | mask_y_3)
    defect_2 = (mask_x_2 | mask_x_4) & (mask_y_2 | mask_y_4)

    defect_mask = defect_1 | defect_2
    mesh_heterogeneous.elemental_fields["RHO"] -= (
        defect_mask[:, None] * 0
    )

    mesh_heterogeneous.elemental_fields["VP"] += (  
        defect_mask[:, None] * 1000
    )






    stf = sn.simple_config.stf.Ricker(center_frequency=CENTRAL_FREQUENCY)

    end_time = 4e-6


    sim_config = sn.UnstructuredMeshSimulationConfiguration(
        unstructured_mesh=mesh_homogeneous,
        name=fr"sc_mesh_homogeneous_{element_per_wavelength}",
        # Event specific configuration.
        event_configuration=sn.EventConfiguration(
            # Source wavelet.
            wavelet=stf,
            waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
                end_time_in_seconds=end_time
            ),
        ),
    )

    # add simulation configuration to Project
    p.add_to_project(
        sim_config, overwrite=True
        )


    simulation_name = fr"mesh_heterogeneous_{element_per_wavelength}"

    sim_config = sn.UnstructuredMeshSimulationConfiguration(

        unstructured_mesh=mesh_heterogeneous,
        name=simulation_name,
        # Event specific configuration.
        event_configuration=sn.EventConfiguration(
            # Source wavelet.
            wavelet=stf,
            waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
                end_time_in_seconds=end_time
            ),
        ),
    )

    # add simulation configuration to Project
    p.add_to_project(
        sim_config, overwrite=True
        )


    dofs =  mesh_heterogeneous.number_of_nodes
    print(f'Start simulation: {simulation_name}')
    print(f'Dofs (number of nodes): {dofs}')


    """
    Launch simulations

    """

    start_time = datetime.now()


    p.simulations.launch(
        ranks_per_job=RANKS_PER_JOB,
        site_name=SALVUS_FLOW_SITE_NAME,
        events=p.events.list(),
        simulation_configuration=simulation_name,
        delete_conflicting_previous_results=True,
        )

    p.simulations.query(block=True)


    time_end = datetime.now()


    execution_time_seconds = (time_end - start_time).total_seconds()
    minutes = int(execution_time_seconds // 60)  # Extract minutes
    seconds = execution_time_seconds % 60  # Extract remaining seconds

    print(f"Execution time: {minutes} minutes and {seconds:.2f} seconds")


# p.viz.nb.simulation_setup(
#     simulation_configuration='mesh_heterogeneous_1',
#     events=p.events.list(),
# )


# # simulation with volume data (full wavefield)
# p.simulations.launch(
#     ranks_per_job=RANKS,
#     site_name=SITE_NAME,
#     events=p.events.list(),
#     simulation_configuration="sc_mesh_homogeneous_scatters",
#     extra_output_configuration={
#         "volume_data": {
#             "sampling_interval_in_time_steps": 20,
#             "fields": ["phi"],
#         },
#     },
#     # We have previously simulated the same event but without
#     # extra output. We have to thus overwrite the existing
#     # simulation.
#     delete_conflicting_previous_results=True,
# )








