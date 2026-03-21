
import numpy as np

import salvus.namespace as sn

from my_code.utilities import *
from pathlib import Path
import os
import salvus
from material_ela_constants.Elastic_Material import Austenite
from salvus.material._details import material as md
import xarray as xr
import salvus.mesh.layered_meshing as lm

from datetime import datetime
import cv2
from PIL import Image

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


N_GRAIN = 5
PROJECT_NAME = fr"tomography_boxdomain_grains_{N_GRAIN}"





# (x0, x1, y0, y1)
domain = (0.0, 0.01, 0.0, 0.01)

box_domain = sn.domain.dim2.BoxDomain(x0=domain[0], x1=domain[1], y0=domain[2], y1=domain[3])
p = sn.Project.from_domain(
    path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=box_domain, load_if_exists=True
)






n_tx = 9
n_rx = 9
edge_gap = 0.02
fields = ['displacement']
f_dir = 'y'

srcs, rxs = Transducers_2D(n_tx=n_tx, n_rx=n_rx, 
                                   edge_gap=edge_gap, domain=domain, f_dir=f_dir,recording_fields=fields).create_salvus_source_receivers()

events = []



# add all receivers to each event of one point source
for i, src in enumerate(srcs):
    events.append(
        sn.Event(event_name=f"event_{i}", sources=src, receivers=rxs)
    )


add_events_to_Project(p, events)






# # Event for simulation
# bdr_gap = 0.02 
# srcs_pos = Vector(x1/8, y1 * (1-bdr_gap))
# dir = (0, 1)
# srcs = VectorPoint2D(x=srcs_pos.x,y=srcs_pos.y, fx=dir[0], fy=dir[1]) 

# n_rxs = 9
# np.linspace(0.1*x1,0.9*x1,n_rxs)

# rxs_pos = [Vector(x, y1 * bdr_gap) for x in np.linspace(0.1*x1,0.9*x1,n_rxs)]


# events = []
# fileds = ["displacement"]     # received fileds

# rxs = [Point2D(x=r.x, y=r.y,
#         station_code=f"REC_{i + 1}",
#         # Note that one can specify the desired recording field here.
#         fields=fileds,)
#     for i, r in enumerate(rxs_pos)
#     ]


# events.append(sn.Event(event_name=f"event_0", sources=srcs, receivers=rxs))


# # add the events to Project
# add_events_to_Project(p, events)




# Import orientation map 

script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "image", f"voronoi_{N_GRAIN}.png")

img = Image.open(img_path)
rgb_img = np.array(img.convert('RGB'))


hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV).astype(np.float32)
hsv_img = hsv_img[::-1, :, :]


orientation_data = xr.DataArray(
    hsv_img[..., 0],  # hue channel
    coords={
        "y": np.linspace(domain[2], domain[3], hsv_img.shape[0]),  # rows → y
        "x": np.linspace(domain[0], domain[1], hsv_img.shape[1])   # cols → x
    },
    dims=["y", "x"]  # order matches (row=y, col=x)
)

# # Plot Histogram of HSV Values
# # Plot a histogram of the HSV image's hue channel using matplotlib
# plt.figure(figsize=(10, 6))
# plt.hist(hsv_img[..., 0].flatten(), bins=25, color='blue', edgecolor='black')
# plt.title('Histogram of HSV Hue Channel')
# plt.xlabel('Hue (angle) Value')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
# orientation_data.plot()


# define material
matl = Austenite()

# hexagonal material (or VTI)
matl_hex = sn.material.elastic.hexagonal.TensorComponents.from_params(**matl.VTI_parameters()) 


# default orientation
orientation_of_grains_default = sn.material.orientation.ClockwiseAngle.from_params(angle_in_degrees=0)

# define orientations
orientation_of_grains = sn.material.orientation.ClockwiseAngle.from_params(angle_in_degrees=orientation_data)

# unorientated material in 2D
material_unorientated = md.to_solver_form(matl_hex.with_orientation(orientation_of_grains_default), ndim=2)

# orientated material with orientation angles in 2D
material_orientated = md.to_solver_form(matl_hex.with_orientation(orientation_of_grains), ndim=2)




'''
Mesh Generation and Simulation
'''


elements_per_wavelength = 5
model_order = 4






# Define the mesh resolution using salvus
mesh_resolution = sn.MeshResolution(
    reference_frequency=CENTRAL_FREQUENCY * 2,  # Reference frequency for the mesh
    elements_per_wavelength=elements_per_wavelength,  # Number of elements per wavelength
    model_order=model_order  # Model order for the mesh
)






# Create the mesh for the grains model using the defined domain and mesh resolution
heterogeneous_model_ab = sn.layered_meshing.MeshingProtocol(
    material_orientated,
    ab=salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
        free_surface=False,
        number_of_wavelengths=1.5,
        reference_velocity=5000,
        reference_frequency=CENTRAL_FREQUENCY * 2,
    ),
)

mesh_grains = sn.layered_meshing.mesh_from_domain(
    domain=box_domain,
    model=heterogeneous_model_ab,
    mesh_resolution=mesh_resolution
)






stf = sn.simple_config.stf.Ricker(center_frequency=CENTRAL_FREQUENCY)

end_time = 4e-6


simulation_name = fr"mesh_heterogeneous_{elements_per_wavelength}"

sim_config = sn.UnstructuredMeshSimulationConfiguration(

    unstructured_mesh=mesh_grains,
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


dofs =  mesh_grains.number_of_nodes
print(f'Start simulation: {simulation_name}')
print(f'Dofs (number of nodes): {dofs}')




# prior model 
matl_hex = sn.material.elastic.hexagonal.Velocity.from_material(matl_hex) 

VP = int(matl_hex.VPV.p)
VS = int(matl_hex.VSV.p)
RHO = int(matl_hex.RHO.p)

homogeneous_model = lm.LayeredModel(
    [
        sn.material.elastic.Velocity.from_params(rho=RHO, vp=VP, vs=VS),
    ]
)

# Create the mesh for the no grains model using the defined domain and mesh resolution
homogeneous_model_ab = sn.layered_meshing.MeshingProtocol(
    homogeneous_model,
    ab=salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
        free_surface=False,
        number_of_wavelengths=1.5,
        reference_velocity=5000,
        reference_frequency=CENTRAL_FREQUENCY * 2,
    ),
)

mesh_no_grains = sn.layered_meshing.mesh_from_domain(
    
    domain=box_domain,
    model=homogeneous_model_ab,
    mesh_resolution=mesh_resolution,
)






# simulation for 
simulation_name_prior = fr"mesh_homogeneous_{elements_per_wavelength}"

sim_config = sn.UnstructuredMeshSimulationConfiguration(

    unstructured_mesh=mesh_no_grains,
    name=simulation_name_prior,
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


centroids = mesh_no_grains.get_element_centroid()


# region of interest for inversion
roi = (centroids[:,0] <= domain[1]) & (centroids[:,0] >= domain[0]) & (centroids[:,1] <= domain[3]) & (centroids[:,1] >= domain[2])

mesh_roi = mesh_no_grains.copy()
ele_fields = [field for field in mesh_roi.elemental_fields]
for field in ele_fields:
    mesh_roi.elemental_fields.pop(field)
mesh_roi.attach_field(
    "region_of_interest",
    np.broadcast_to(roi[:, None], mesh_roi.connectivity.shape),
)




# p.viz.nb.simulation_setup(
#     simulation_configuration=simulation_name,
#     events=p.events.list(),
# )



# """
# Launch simulations

# """

# start_time = datetime.now()


# p.simulations.launch(
#     ranks_per_job=RANKS_PER_JOB,
#     site_name=SALVUS_FLOW_SITE_NAME,
#     events=p.events.list(),
#     simulation_configuration=simulation_name,
#     delete_conflicting_previous_results=True,
#     )


# # simulation with volume data (full wavefield)
# p.simulations.launch(
#     ranks_per_job=RANKS_PER_JOB,
#     site_name=SALVUS_FLOW_SITE_NAME,
#     events=p.events.list()[0],
#     simulation_configuration=simulation_name,
#     extra_output_configuration={
#         "volume_data": {
#             "sampling_interval_in_time_steps": 20,
#             "fields": ["displacement"],
#         },
#     },
#     # We have previously simulated the same event but without
#     # extra output. We have to thus overwrite the existing
#     # simulation.
#     delete_conflicting_previous_results=True,
# )




# p.simulations.query(block=True)


# time_end = datetime.now()


# execution_time_seconds = (time_end - start_time).total_seconds()
# minutes = int(execution_time_seconds // 60)  # Extract minutes
# seconds = execution_time_seconds % 60  # Extract remaining seconds

# print(f"Execution time: {minutes} minutes and {seconds:.2f} seconds")




# misfitname = 'L2_misfit_for_hetero_model'

# # The misfit configuration defines how synthetics are compared to observed data.
# p += sn.MisfitConfiguration(
#     name=misfitname,
#     # Could be observed data. Here we compare to the synthetic target.
#     observed_data=simulation_name,
#     # Salvus comes with a variety of misfit functions. You can
#     # also define your own.
#     misfit_function="L2",
#     # This is an acoustic simulation so we'll use recordings of phi.
#     receiver_field=fields[0],
# )

# event_names_imaging = p.events.list()





# invconfig = sn.InverseProblemConfiguration(
#     name="inversion_L2",
#     # Starting model is the model without scatterers.
#     prior_model=simulation_name_prior,
#     # The events to use.
#     events=event_names_imaging,
#     misfit_configuration=misfitname,
#     # What parameters to invert for.
#     mapping=sn.Mapping(
#         scaling="absolute",
#         inversion_parameters=["VP", "VS"],
#         region_of_interest=mesh_roi,
#         # postprocess_model_update = tensor_to_orientation
#     ),
#     # preconditioner=sn.ConstantSmoothing({"VP": 0.5e-3}),
    
#     # The inversion method and its settings.
#     method=sn.TrustRegion(initial_trust_region_linf=10.0),
#     # The misfit configuration we defined above.
    
#     # Compress the forward wavefield by subsampling in time.
#     wavefield_compression=sn.WavefieldCompression(
#         forward_wavefield_sampling_interval=10
#     ),
#     # Job submission settings.
#     job_submission=sn.SiteConfig(site_name=SALVUS_FLOW_SITE_NAME, ranks_per_job=RANKS_PER_JOB),
# )

# add_inversion(p, invconfig)


# p.inversions.set_job_submission_configuration(
#     "inversion_L2", sn.SiteConfig(site_name=SALVUS_FLOW_SITE_NAME, ranks_per_job=RANKS_PER_JOB)
# )

# # Lastly we perform two iterations, and have a look at the results.
# for i in range(10):
    
#     p.inversions.iterate(
#         inverse_problem_configuration="inversion_L2",
#         timeout_in_seconds=360,
#         ping_interval_in_seconds=50,
#         delete_disposable_files="all",
#     )
    