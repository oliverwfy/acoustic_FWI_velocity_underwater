
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




SITE_NAME = "oliver_wsl"
RANKS = 8

# 1 MHz should run on laptops, 4 MHz and higher we recommend GPUs
CENTRAL_FREQUENCY = 1.5e6  # MHz

assert CENTRAL_FREQUENCY >= 1.5e6

PROJECT_NAME = f"tomography_heterogeneous_n"


x0, x1 = 0.0, 0.01
y0, y1 = 0.0, 0.01

domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)

p = sn.Project.from_domain(
    path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True
)

# acoustic material
homogeneous_model = lm.LayeredModel(
    [
        sn.material.acoustic.Velocity.from_params(rho=2600.0, vp=5000.0),
    ]
)

homogeneous_model_ab = sn.layered_meshing.MeshingProtocol(
    homogeneous_model,
    ab=salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
        free_surface=['y0','y1'],
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
x_within_range = (centroids[:, 0] >= 0) & (centroids[:, 0] <= 10e-3)
y_within_range = (centroids[:, 1] >= 4e-3) & (centroids[:, 1] <= 6e-3)

selected_ele = y_within_range & x_within_range
mesh_homogeneous_scatterers.elemental_fields["RHO"] -= (
    selected_ele[:, None] * 0
)
mesh_homogeneous_scatterers.elemental_fields["VP"] += (  
    selected_ele[:, None] * 2000
)


# region of interest for inversion
roi = (
    mesh_homogeneous_scatterers.get_element_centroid()[..., 1]
    <= 9e-3
) &  (
    mesh_homogeneous_scatterers.get_element_centroid()[..., 1]
    > 1e-3
)

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
    sim_config, overwrite=False
    )


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
    sim_config, overwrite=False
    )







nx = 9

array_0 = ArrayTransducer2D(nx=nx, x0=0.001, dx=0.009 / nx, array_name="array_0", source_y=[0, 0.01])

assert array_0.test_within_domain(domain), "array_0 is not within the domain"


source_id_arr = np.arange(0, nx, 1)


# Now we define the actual inverse problem
events = []
source_y_ls = array_0.source_y
for source_y in source_y_ls:
    for source_id in source_id_arr:
        source, receivers = array_0.create_salvus_source_receivers(
            source_index=source_id, source_y = source_y
        )
        events.append(sn.Event(
            event_name=f"array_{source_y}_source_{source_id}",
            sources=source,
            receivers=receivers,
        ))

add_events_to_Project(p, events)






p.viz.nb.simulation_setup(
    simulation_configuration='sc_mesh_homogeneous_scatters',
    events=p.events.list(),
)


# p.simulations.launch(
#     simulation_configuration="sc_mesh_homogeneous_scatters",
#     events=p.events.list(),
#     site_name=SITE_NAME,
#     ranks_per_job=RANKS,
# )


# p.simulations.query(block=True)


# # The misfit configuration defines how synthetics are compared to observed data.
# p += sn.MisfitConfiguration(
#     name="L2_misfit_for_homo_model",
#     # Could be observed data. Here we compare to the synthetic target.
#     observed_data="sc_mesh_homogeneous_scatters",
#     # Salvus comes with a variety of misfit functions. You can
#     # also define your own.
#     misfit_function="L2",
#     # This is an acoustic simulation so we'll use recordings of phi.
#     receiver_field="phi",
# )



# event_names_imaging = p.events.list()


# invconfig = sn.InverseProblemConfiguration(
#     name="inversion_L2",
#     # Starting model is the model without scatterers.
#     prior_model="sc_mesh_homogeneous",
#     # The events to use.
#     events=event_names_imaging,
#     misfit_configuration="L2_misfit_for_homo_model",
#     # What parameters to invert for.
#     mapping=sn.Mapping(
#         scaling="relative_deviation_from_prior",
#         inversion_parameters=["VP"],
#         region_of_interest=mesh_roi,
#         # postprocess_model_update = tensor_to_orientation
#     ),
#     # The inversion method and its settings.
#     method=sn.TrustRegion(initial_trust_region_linf=1.0),
#     # The misfit configuration we defined above.
    
#     # Compress the forward wavefield by subsampling in time.
#     wavefield_compression=sn.WavefieldCompression(
#         forward_wavefield_sampling_interval=10
#     ),
#     # Job submission settings.
#     job_submission=sn.SiteConfig(site_name=SITE_NAME, ranks_per_job=RANKS),
# )

# add_inversion(p, invconfig)



# p.inversions.set_job_submission_configuration(
#     "inversion_L2", sn.SiteConfig(site_name=SITE_NAME, ranks_per_job=RANKS)
# )

# # Lastly we perform two iterations, and have a look at the results.
# for i in range(10):
    
#     p.inversions.iterate(
#         inverse_problem_configuration="inversion_L2",
#         timeout_in_seconds=360,
#         ping_interval_in_seconds=10,
#         delete_disposable_files="all",
#     )
    

