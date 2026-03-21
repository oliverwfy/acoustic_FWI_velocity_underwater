
import numpy as np

import salvus.namespace as sn

import salvus.mesh
import salvus.mesh.layered_meshing as lm
from my_code.utilities import *
from pathlib import Path
import salvus.project as sp
from salvus.material import elastic, orientation
from salvus import material


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




SITE_NAME = "oliver_wsl"
RANKS = 8

# 1 MHz should run on laptops, 4 MHz and higher we recommend GPUs
CENTRAL_FREQUENCY = 1e6  # MHz

assert CENTRAL_FREQUENCY >= 1e6

PROJECT_NAME = f"tomography_heterogeneous_box"




x0, x1 = 0.0, 0.01
y0, y1 = 0.0, 0.01

domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)
p = sn.Project.from_domain(
    path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True
)



# homogeneous_model = lm.LayeredModel(
#     [
#         sn.material.elastic.Velocity.from_params(rho=2600.0, vp=5000.0, vs=3000),
#     ]
# )

homogeneous_model = lm.LayeredModel(
    [
        sn.material.acoustic.Velocity.from_params(rho=2600.0, vp=5000.0),
    ]
)

homogeneous_model_ab = sn.layered_meshing.MeshingProtocol(
    homogeneous_model,
    # ab=salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
    #     free_surface=['t0', 't1', 'r0'],
    #     number_of_wavelengths=1.5,
    #     reference_velocity=5000,
    #     reference_frequency=CENTRAL_FREQUENCY * 2,
    # ),
)

mesh_homogeneous = lm.mesh_from_domain(
    domain=domain,
    model=homogeneous_model_ab,
    mesh_resolution=sn.MeshResolution(
        reference_frequency=CENTRAL_FREQUENCY * 2, elements_per_wavelength=2
    ),
)

dofs =  mesh_homogeneous.number_of_nodes
print(f'Dofs (number of nodes): {dofs}')







# orientation_data = map_orientation(voronoi.png)
# orientation_of_grains = sn.material.orientation.ClockwiseAngle.from_params(angle_in_degrees=orientation_data)

material_unoriented = sn.material.elastic.hexagonal.TensorComponents.from_params(**matl.VTI_parameters())


material_oriented = material_unoriented.with_orientation(orientation_of_grains)

# Convert the oriented material to solver form for 2D simulations
material1 = md.to_solver_form(material_oriented, ndim=2)

# Define a default orientation for comparison
orientation_of_grains_default = sn.material.orientation.ClockwiseAngle.from_params(angle_in_degrees=0)

# Convert the unoriented material with default orientation to solver form for 2D simulations
material2 = md.to_solver_form(material_unoriented.with_orientation(orientation_of_grains_default), ndim=2)



np.set_printoptions(precision=2, suppress=True)  # To avoid long outputs


from salvus.material._details.material import to_solver_form
orientation_1 = orientation.AzimuthDip.from_params(azimuth=0, dip=45)

orientated_matl = to_solver_form(material_unoriented.with_orientation(orientation_1))
sn.material.utils.extract_tensor(orientated_matl)



# # Make a copy to add strong scatterers after meshing
# mesh_homogeneous_scatterers = mesh_homogeneous.copy()
# for scatterer_center in [np.array([x, 0.015]) for x in [0.015]]:
#     distance_from_scatterer = np.linalg.norm(
        
#         mesh_homogeneous_scatterers.get_element_centroid() - scatterer_center,
#         axis=1,
#     )
#     radius = 0.005
    
#     anomaly = distance_from_scatterer < radius
#     mesh_homogeneous_scatterers.elemental_fields["RHO"] -= (
#         anomaly[:, None] * 0
#     )
#     mesh_homogeneous_scatterers.elemental_fields["VP"] -= (
#         anomaly[:, None] * 1000
#     )

# Make a copy to add a layer in between
mesh_homogeneous_scatterers = mesh_homogeneous.copy()
centroids = mesh_homogeneous_scatterers.get_element_centroid()
defect_mask = np.linalg.norm(centroids, axis=1) < 4e-3
mesh_homogeneous_scatterers.elemental_fields["RHO"] -= (
    defect_mask[:, None] * 0
)
mesh_homogeneous_scatterers.elemental_fields["VP"] += (  
    defect_mask[:, None] * 1000
)

# mesh_homogeneous_scatterers.elemental_fields["VS"] += (  
#     defect_mask[:, None] * 1000
# )


# region of interest for inversion
roi = np.linalg.norm(centroids, axis=1) < 8e-3

mesh_roi = mesh_homogeneous_scatterers.copy()
fields = [field for field in mesh_roi.elemental_fields]
for field in fields:
    mesh_roi.elemental_fields.pop(field)
mesh_roi.attach_field(
    "region_of_interest",
    np.broadcast_to(roi[:, None], mesh_roi.connectivity.shape),
)




stf = sn.simple_config.stf.Ricker(center_frequency=CENTRAL_FREQUENCY)

end_time = 4e-6


sim_config = sn.UnstructuredMeshSimulationConfiguration(
    unstructured_mesh=mesh_homogeneous,
    name="sc_mesh_homogeneous",
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
    name="sc_mesh_homogeneous_scatters",
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


# def locate_source_in_mesh(n_txs):
#     angles_deg = np.linspace(0, 360, n_txs + 1)[:-1]  
#     angles_rad = np.deg2rad(angles_deg)
#     x = np.cos(angles_rad)
#     y = np.sin(angles_rad)
#     nomal_vec = np.round(np.stack((x, y), axis=1), 6)
#     tx_coordinates = nomal_vec * radius

#     mesh_nodes = mesh_homogeneous.get_element_nodes().reshape(-1, 2)
#     unique_mesh_nodes = np.unique(mesh_nodes, axis=0)

#     closest_node = np.array([
#         unique_mesh_nodes[np.argmin(np.linalg.norm(unique_mesh_nodes - tx, axis=1))]
#         for tx in tx_coordinates
#     ])
#     return tx_coordinates, nomal_vec

# # Generate txs and txs
# # number of sources
# n_txs = 8
# tx_coordinates, nomal_vec = locate_source_in_mesh(n_txs=n_txs)
# sources = [sn.simple_config.source.cartesian.VectorPoint2D(x=tx[0], y=tx[1], fx=nomal_vec[_i,0], fy=nomal_vec[_i,1]) for _i,tx in enumerate(tx_coordinates)] 


# n_rxs = 32

# rx_coordinates, _ = locate_source_in_mesh(n_txs=n_rxs)
# fileds = ["displacement"]

# receivers = [
#     sn.simple_config.receiver.seismology.SideSetPoint2D(
#         longitude= l, depth_in_m= 0, radius_of_sphere_in_m =radius ,
#         side_set_name = ['r1'],
#         fields=fileds,
#         station_code=f"{_i:06d}",
#     )
#     for _i, l in enumerate(np.linspace(0, 360, n_txs + 1)[:-1]  )
# ]

n_txs = 8
angles_deg_tx = np.linspace(0, 360, n_txs + 1)[:-1]  
# sources = [
#     sn.simple_config.source.seismology.SideSetVectorPoint2D(
#         longitude=l,
#         depth_in_m=0.001,
#         radius_of_sphere_in_m=radius,
#         fr=1,  # radial force
#         fp=0,  # polar force
#         side_set_name="r1"
#     )
#     for l in angles_deg_tx
# ]

sources = [
    sn.simple_config.source.seismology.SideSetScalarPoint2D(
        longitude=l,
        depth_in_m=0.001,
        radius_of_sphere_in_m=radius,
        f=1,
        side_set_name="r1"
    )
    for l in angles_deg_tx
]

n_rxs = 32
angles_deg_rx = np.linspace(0, 360, n_rxs + 1)[:-1]  
fileds = ["phi"]

receivers = [
    sn.simple_config.receiver.seismology.SideSetPoint2D(
        longitude=l,
        depth_in_m=0.001,
        radius_of_sphere_in_m=radius,
        side_set_name="r1",
        fields=fileds,
        station_code=f"{_i:06d}",
    )
    for _i,l in enumerate(angles_deg_rx)
]
 
events = [sn.Event(event_name=f"event_{i}", sources=s, receivers=receivers) for i,s in enumerate(sources)]

add_events_to_Project(p, events)






p.viz.nb.simulation_setup(
    simulation_configuration='sc_mesh_homogeneous_scatters',
    events=p.events.list(),
)


# simulation with volume data (full wavefield)
p.simulations.launch(
    ranks_per_job=RANKS,
    site_name=SITE_NAME,
    events=p.events.list(),
    simulation_configuration="sc_mesh_homogeneous_scatters",
    extra_output_configuration={
        "volume_data": {
            "sampling_interval_in_time_steps": 20,
            "fields": ["phi"],
        },
    },
    # We have previously simulated the same event but without
    # extra output. We have to thus overwrite the existing
    # simulation.
    delete_conflicting_previous_results=True,
)


# p.simulations.launch(
#     simulation_configuration="sc_mesh_homogeneous_scatters",
#     events=p.events.list(),
#     site_name=SITE_NAME,
#     ranks_per_job=RANKS,
# )


p.simulations.query(block=True)


# The misfit configuration defines how synthetics are compared to observed data.
p += sn.MisfitConfiguration(
    name="L2_misfit_for_homo_model",
    # Could be observed data. Here we compare to the synthetic target.
    observed_data="sc_mesh_homogeneous_scatters",
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
    prior_model="sc_mesh_homogeneous",
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
    # preconditioner=sn.ConstantSmoothing({"VP": 1e-3, "VS": 1e-3}),
    # The inversion method and its settings.
    method=sn.TrustRegion(initial_trust_region_linf= 0.1),
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


for i in range(10):
    
    p.inversions.iterate(
        inverse_problem_configuration="inversion_L2",
        timeout_in_seconds=360,
        ping_interval_in_seconds=50,
        delete_disposable_files="all",
    )
    
