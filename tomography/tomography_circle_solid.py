
import numpy as np

import salvus.namespace as sn

import salvus.mesh
import salvus.mesh.layered_meshing as lm
from my_code.utilities import *
from pathlib import Path



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



# salvus site 
SITE_NAME = "oliver_wsl"
RANKS = 8

# number of grains
N_GRAIN = 5

# center frequency 
CENTRAL_FREQUENCY = 2e6  # MHz
# CENTRAL_FREQUENCY = 3e6  # MHz


roi_radius = 9e-3

n_txs = 32


VP = 5000.0
VS = 3000.0
RHO = 2600.0


x0, x1 = 0.0, 0.02
y0, y1 = 0.0, 0.02

time_ratio = 3
end_time = 2e-6 * time_ratio

smoothing = 0

# project name (folder's name)
# PROJECT_NAME = fr"tomography_solid_heterogeneous_grain_{N_GRAIN}_smooth_{smoothing}"
PROJECT_NAME = fr"tomography_solid_heterogeneous_roi_{int(roi_radius*1e3)}mm_tx_{n_txs}_smooth_{smoothing}_maxtime_{int(time_ratio)}"



domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)

p = sn.Project.from_domain(
    path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True
)

# acoustic material
homogeneous_model = lm.LayeredModel(
    [
        sn.material.elastic.Velocity.from_params(rho=RHO, vp=VP, vs=VS),
    ] 
)


homogeneous_model_ab = sn.layered_meshing.MeshingProtocol(
    homogeneous_model,
    ab=salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
        free_surface=False,
        number_of_wavelengths=1.5,
        reference_velocity=5000,
        reference_frequency=CENTRAL_FREQUENCY * 2,
    ),
)

mesh_homogeneous = lm.mesh_from_domain(
    domain=domain,
    model=homogeneous_model_ab,
    mesh_resolution=sn.MeshResolution(
        reference_frequency=CENTRAL_FREQUENCY * 2, elements_per_wavelength=3
    ),
)






IMAGE_PATH = get_script_dir() / "image" / f"voronoi_{N_GRAIN}.png"


indexer = VoronoiGrainIndexer(IMAGE_PATH, min_percent=0.1, dilate_iters=1)
indexer.process()





dofs =  mesh_homogeneous.number_of_nodes
print(f'Dofs (number of nodes): {dofs}')


# Make a copy to add a layer in between
mesh_homogeneous_scatterers = mesh_homogeneous.copy()
centroids = mesh_homogeneous_scatterers.get_element_centroid()
roi_circle_mask = np.linalg.norm(centroids - np.array([x1/2, y1/2]), axis=1) < 5e-3
centroids -= [1/4*x1, 1/4*y1]
centroids /= x1/2
roi_box_mask = (centroids[:,0] >=0) & (centroids[:,0] <=1.0) & (centroids[:,1] >=0) & (centroids[:,1] <=1.0)

roi_mask = roi_circle_mask & roi_box_mask
ele_inside = np.round(centroids[roi_mask], 6)


grain_mask_full = np.zeros(len(centroids), dtype=int)  # default 0

grain_mask = np.array([indexer.point_to_grain_id(x=ele[0], y=ele[1]) for ele in ele_inside]) + 1

grain_mask_full[roi_mask] = grain_mask


# mesh_homogeneous_scatterers.elemental_fields["RHO"] -= (
#     defect_mask[:, None] * 0
# )
mesh_homogeneous_scatterers.elemental_fields["VP"] += (  
    (grain_mask_full[:,None]) * 100
)

# mesh_homogeneous_scatterers.elemental_fields["VS"] += (  
#     defect_mask[:, None] * 1000
# )





source_r = 9e-3
center = np.array([x1/2, y1/2])


def locate_source_in_mesh(n_txs, center, radius):
    angles_deg = np.linspace(0, 360, n_txs + 1)[:-1]  
    angles_rad = np.deg2rad(angles_deg)
    x = np.cos(angles_rad)
    y = np.sin(angles_rad)
    nomal_vec = np.round(np.stack((x, y), axis=1), 6)
    tx_coordinates = nomal_vec * radius + center

    mesh_nodes = mesh_homogeneous.get_element_nodes().reshape(-1, 2)
    unique_mesh_nodes = np.unique(mesh_nodes, axis=0)

    closest_node = np.array([
        unique_mesh_nodes[np.argmin(np.linalg.norm(unique_mesh_nodes - tx, axis=1))]
        for tx in tx_coordinates
    ])
    return tx_coordinates, nomal_vec

# Generate txs and txs
# number of sources
tx_coordinates, nomal_vec = locate_source_in_mesh(n_txs=n_txs, center = center,radius=source_r)
mag_ratio = 1e12
sources = [
    sn.simple_config.source.cartesian.VectorPoint2D(
        x=tx[0], y=tx[1], 
        fx=nomal_vec[_i,0]*mag_ratio, fy=nomal_vec[_i,1]*mag_ratio
        ) 

    for _i, tx in enumerate(tx_coordinates)
] 


n_rxs = 32

rx_coordinates, _ = locate_source_in_mesh(n_txs=n_rxs, center = center, radius=source_r)
fileds = ["displacement"]

events = []


for _i, src in enumerate(sources): 
    receivers = [
    sn.simple_config.receiver.cartesian.Point2D(
        x=rx[0], y=rx[1], 
        fields=fileds, station_code=f"{_i:06d}",
        ) 
    for _i, rx in enumerate(rx_coordinates) if  np.linalg.norm(src.location - rx) > source_r / 2

]
    
    events.append(sn.Event(event_name=f"event_{_i}", sources=src, receivers=receivers))


# n_rxs = 32

# rx_coordinates, _ = locate_source_in_mesh(n_txs=n_rxs, center = center,radius=source_r)
# fileds = ["displacement"]

# receivers = [
#     sn.simple_config.receiver.cartesian.Point2D(
#         x=rx[0], y=rx[1], 
#         fields=fileds, station_code=f"{_i:06d}",
#         ) 
#     for _i, rx in enumerate(rx_coordinates)
# ]


# events = [sn.Event(event_name=f"event_{i}", sources=s, receivers=receivers) for i,s in enumerate(sources)]

add_events_to_Project(p, events)







# region of interest for inversion
centroids = mesh_homogeneous_scatterers.get_element_centroid()
roi = np.linalg.norm(centroids - np.array([x1/2, y1/2]), axis=1) < roi_radius

mesh_roi = mesh_homogeneous_scatterers.copy()
fields = [field for field in mesh_roi.elemental_fields]
for field in fields:
    mesh_roi.elemental_fields.pop(field)
mesh_roi.attach_field(
    "region_of_interest",
    np.broadcast_to(roi[:, None], mesh_roi.connectivity.shape),
)





stf = sn.simple_config.stf.Ricker(center_frequency=CENTRAL_FREQUENCY)


sim_config = sn.UnstructuredMeshSimulationConfiguration(
    unstructured_mesh=mesh_homogeneous,
    name="mesh_homogeneous",
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



sim_config = sn.UnstructuredMeshSimulationConfiguration(
    unstructured_mesh=mesh_homogeneous_scatterers,
    name="mesh_heterogeneous",
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



p.viz.nb.simulation_setup(
    simulation_configuration='mesh_heterogeneous',
    events=p.events.list(),
)





# # # simulation with volume data (full wavefield)
# # p.simulations.launch(
# #     ranks_per_job=RANKS,
# #     site_name=SITE_NAME,
# #     events=p.events.list(),
# #     simulation_configuration="mesh_heterogeneous",
# #     extra_output_configuration={
# #         "volume_data": {
# #             "sampling_interval_in_time_steps": 20,
# #             "fields": ["phi"],
# #         },
# #     },
# #     # We have previously simulated the same event but without
# #     # extra output. We have to thus overwrite the existing
# #     # simulation.
# #     delete_conflicting_previous_results=True,
# # )








p.simulations.launch(
    simulation_configuration="mesh_heterogeneous",
    events=p.events.list(),
    site_name=SITE_NAME,
    ranks_per_job=RANKS,
)


p.simulations.query(block=True)


# The misfit configuration defines how synthetics are compared to observed data.
p += sn.MisfitConfiguration(
    name="L2_misfit_for_homo_model",
    # Could be observed data. Here we compare to the synthetic target.
    observed_data="mesh_heterogeneous",
    # Salvus comes with a variety of misfit functions. You can
    # also define your own.
    misfit_function="L2",
    # This is an acoustic simulation so we'll use recordings of phi.
    receiver_field=fileds[0],
)



event_names_imaging = p.events.list()


invconfig = sn.InverseProblemConfiguration(
    name="inversion_L2",
    # Starting model is the model without scatterers.
    prior_model="mesh_homogeneous",
    # The events to use.
    events=event_names_imaging,
    misfit_configuration="L2_misfit_for_homo_model",
    # What parameters to invert for.
    mapping=sn.Mapping(
        scaling="relative_deviation_from_prior",
        inversion_parameters=["VP"],
        region_of_interest=mesh_roi,
        # postprocess_model_update = tensor_to_orientation
    ),
    # preconditioner=sn.ConstantSmoothing({"VP": 0.5e-3}),
    # preconditioner=sn.ConstantSmoothing({"VP": smoothing*1e-3}),

    # The inversion method and its settings.
    method=sn.TrustRegion(initial_trust_region_linf=1 ),
    # The misfit configuration we defined above.
    
    # Compress the forward wavefield by subsampling in time.
    wavefield_compression=sn.WavefieldCompression(
        forward_wavefield_sampling_interval=10
    ),
    # Job submission settings.
    job_submission=sn.SiteConfig(site_name=SITE_NAME, ranks_per_job=RANKS),
)

add_inversion(p, invconfig)



p.inversions.set_job_submission_configuration(
    "inversion_L2", sn.SiteConfig(site_name=SITE_NAME, ranks_per_job=RANKS)
)

# Lastly we perform two iterations, and have a look at the results.
for i in range(10):
    
    p.inversions.iterate(
        inverse_problem_configuration="inversion_L2",
        timeout_in_seconds=500,
        ping_interval_in_seconds=50,
        delete_disposable_files="all",
    )
    

