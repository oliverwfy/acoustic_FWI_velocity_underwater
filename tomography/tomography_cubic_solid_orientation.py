
import numpy as np

import salvus.namespace as sn

import salvus.mesh
import salvus.mesh.layered_meshing as lm
from my_code.utilities import *
from pathlib import Path
from material_ela_constants.Elastic_Material import * 


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
CENTRAL_FREQUENCY = 5e5 # MHz
f_c = 2e5
sampling_rate_in_hertz = CENTRAL_FREQUENCY *200
time_step_in_seconds = 1/sampling_rate_in_hertz


scatter_length = 2e-3
roi_radius = 7e-4
mag_ratio = 1e12

n_rxs = 1
n_txs = 1


x0, x1 = 0.0, 0.01
y0, y1 = 0.0, 0.01

time_ratio = 1
end_time = 2e-6 * time_ratio

smoothing = 0

# project name (folder's name)
# PROJECT_NAME = fr"tomography_solid_heterogeneous_grain_{N_GRAIN}_smooth_{smoothing}"
PROJECT_NAME = fr"tomography_cubic_solid_heterogeneous_orientation"



domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)

p = sn.Project.from_domain(
    path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True
)



matl = Titanium()
material_unoriented = sn.material.elastic.hexagonal.TensorComponents.from_params(**matl.VTI_parameters())
from salvus.material._details import material as md
# Define a default orientation for comparison
orientation_of_grains_default = sn.material.orientation.ClockwiseAngle.from_params(angle_in_degrees=np.rad2deg(0))

# Convert the unoriented material with default orientation to solver form for 2D simulations
matl_background = md.to_solver_form(material_unoriented.with_orientation(orientation_of_grains_default), ndim=2)


# acoustic material
homogeneous_model = lm.LayeredModel(
    [
        matl_background,
    ] 
)


# homogeneous_model_ab = sn.layered_meshing.MeshingProtocol(
#     homogeneous_model,
#     ab=salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
#         free_surface=False,
#         number_of_wavelengths=1.5,
#         reference_velocity=5000,
#         reference_frequency=CENTRAL_FREQUENCY * 2,
#     ),
# )

mesh_homogeneous = lm.mesh_from_domain(
    domain=domain,
    model=homogeneous_model,
    mesh_resolution=sn.MeshResolution(
        reference_frequency=CENTRAL_FREQUENCY * 2, elements_per_wavelength=4, model_order=4
    ),
)




orientation_of_grains = sn.material.orientation.ClockwiseAngle.from_params(angle_in_degrees=np.rad2deg(np.pi/6))
material_oriented = md.to_solver_form(material_unoriented.with_orientation(orientation_of_grains), ndim=2)





mesh_homogeneous_scatterers = mesh_homogeneous.copy()
centroids = mesh_homogeneous_scatterers.get_element_centroid()
center = np.array([x1/2, y1/2])
np.abs(centroids - center)
roi_circle_mask = np.max(np.abs(centroids - center), axis=1) <= scatter_length



for key in mesh_homogeneous_scatterers.element_nodal_fields.keys():
    mesh_homogeneous_scatterers.elemental_fields[key][roi_circle_mask] = getattr(material_oriented, key).p






def tx_rx_pos(n, x1, y1, tol=0):
    
    pos = {'x0': [(0+tol, y) for y in np.linspace(0, y1, n+2)[1:-1]],
           'x1': [(x1-tol, y) for y in np.linspace(0, y1, n+2)[1:-1]],
           'y0': [(x, 0+tol) for x in np.linspace(0, x1, n+2)[1:-1]],
           'y1': [(x, y1-tol) for x in np.linspace(0, x1, n+2)[1:-1]]}
    
    return pos



def generate_source(tx_pos, mag_ratio):
    source_ls = []
    fx, fy = 1, 1
    for bdry in tx_pos.keys():
        if 'x' in bdry:
            fy=0
        elif 'y' in bdry:
            fx=0
        source_ls += [
            sn.simple_config.source.cartesian.VectorPoint2D(
                x= t[0], y= t[1],
                fx=fx*mag_ratio, fy=fy*mag_ratio
                )
            for _i, t in enumerate(tx_pos[bdry])
            ]  
    return source_ls

def generate_receivers(rx_pos ,fileds):
    rxs_ls = []
    for bdry in rx_pos.keys():
        rxs_ls += [
            sn.simple_config.receiver.cartesian.Point2D(
                x=r[0], y=r[1], 
                fields=fileds, station_code= bdry + f"_{_i:06d}",
            )
        for _i, r in enumerate(rx_pos[bdry])
        ]

    return rxs_ls


def generete_events(source_ls, rxs_ls):
    events = []
    for _i, src in enumerate(source_ls): 
        events.append(sn.Event(event_name=f"event_{_i}", sources=src, receivers=rxs_ls))
    return events


def roi_circle(mesh, x1, y1, r):
    mesh_roi = mesh.copy()
    centroids = mesh.get_element_centroid()
    roi = np.linalg.norm(centroids - np.array([x1/2, y1/2]), axis=1) < r
    fields = [field for field in mesh_roi.elemental_fields]
    for field in fields:
        mesh_roi.elemental_fields.pop(field)
    mesh_roi.attach_field(
        "region_of_interest",
        np.broadcast_to(roi[:, None], mesh_roi.connectivity.shape),
    )
    return mesh_roi

tx_pos = tx_rx_pos(n_txs, x1, y1)
rx_pos = tx_rx_pos(n_rxs, x1, y1)

source_ls = generate_source(tx_pos, mag_ratio)

fields = ["displacement"]
rxs_ls = generate_receivers(rx_pos, fields)

events = generete_events(source_ls, rxs_ls)

add_events_to_Project(p, events)

# source time function
stf = sn.simple_config.stf.Ricker(center_frequency=f_c)


# Region of interest
mesh_roi = roi_circle(mesh_homogeneous, x1, y1, roi_radius)

sim_config = sn.UnstructuredMeshSimulationConfiguration(
    unstructured_mesh=mesh_homogeneous,
    name="mesh_homogeneous",
    # Event specific configuration.
    event_configuration=sn.EventConfiguration(
        # Source wavelet.
        wavelet=stf,
        waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
            start_time_in_seconds=0,
            end_time_in_seconds=end_time,
            time_step_in_seconds=time_step_in_seconds
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
            start_time_in_seconds=0,
            end_time_in_seconds=end_time,
            time_step_in_seconds=time_step_in_seconds
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





ed = p.waveforms.get(data_name="mesh_heterogeneous", events=p.events.list()[0])[0]
data_true = ed.get_waveform_data_xarray(receiver_field='displacement')
time = data_true.time
ed = p.waveforms.get(data_name="mesh_homogeneous", events=p.events.list()[0])[0]
data_init = ed.get_waveform_data_xarray(receiver_field='displacement')
time = data_init.time


id = 2
f_adj_0 =  (data_true[id] - data_init[id]) / data_true.max() 

fig, axs = plt.subplots(1, 3, figsize=(20, 6), dpi=100)


axs[0].plot(time*1e6, data_true[id]/data_true.max())
axs[0].plot(time*1e6, data_init[id]/data_true.max())
axs[0].legend([r'$u_{obs}$', 'u'])
axs[0].set_xlabel('time (us)')
axs[0].set_ylabel(r'displacement $u_2$')
axs[0].set_title('observed and synthetic data')


axs[1].plot(time*1e6, f_adj_0, color='gray',linestyle='--')
axs[1].set_xlabel('time (us)')
axs[1].set_ylabel(r'displacement $u_2$')
axs[1].set_title('residuals')


axs[2].plot(time*1e6, -np.flip(f_adj_0), color='gray',linestyle='--')
axs[2].set_xlabel('time (us)')
axs[2].set_ylabel(r'displacement $u_2$')
axs[2].set_title('time-reversed residuals')



# p.simulations.launch(
#     simulation_configuration="mesh_heterogeneous",
#     events=p.events.list(),
#     site_name=SITE_NAME,
#     ranks_per_job=RANKS,
# )


# p.simulations.launch(
#     simulation_configuration="mesh_homogeneous",
#     events=p.events.list(),
#     site_name=SITE_NAME,
#     ranks_per_job=RANKS,
# )


# # simulation with volume data (full wavefield)
# p.simulations.launch(
#     ranks_per_job=RANKS,
#     site_name=SITE_NAME,
#     events=p.events.list(),
#     simulation_configuration="mesh_heterogeneous",
#     extra_output_configuration={
#         "volume_data": {
#             "sampling_interval_in_time_steps": 5,
#             "fields": ["displacement"],
#         },
#     },
#     # We have previously simulated the same event but without
#     # extra output. We have to thus overwrite the existing
#     # simulation.
#     delete_conflicting_previous_results=True,
# )




p.simulations.query(block=True)


# # The misfit configuration defines how synthetics are compared to observed data.
# p += sn.MisfitConfiguration(
#     name="L2_misfit",
#     # Could be observed data. Here we compare to the synthetic target.
#     observed_data="mesh_heterogeneous",
#     # Salvus comes with a variety of misfit functions. You can
#     # also define your own.
#     misfit_function="L2",
#     # This is an acoustic simulation so we'll use recordings of phi.
#     receiver_field=fileds[0],
# )



# event_names_imaging = p.events.list()


# invconfig = sn.InverseProblemConfiguration(
#     name="inversion_L2",
#     # Starting model is the model without scatterers.
#     prior_model="mesh_homogeneous",
#     # The events to use.
#     events=event_names_imaging,
#     misfit_configuration="L2_misfit",
#     # What parameters to invert for.
#     mapping=sn.Mapping(
#         scaling="absolute",
#         inversion_parameters=["C11", "C12","C13", "C22","C23", "C33"],
#         region_of_interest=mesh_roi,
#         source_cutout_radius_in_meters=3e-3,
#         receiver_cutout_radius_in_meters=3e-3
#         # postprocess_model_update = tensor_to_orientation
#     ),
#     # preconditioner=sn.ConstantSmoothing({"VP": 0.5e-3}),
#     # preconditioner=sn.ConstantSmoothing({"VP": smoothing*1e-3}),

#     # The inversion method and its settings.
#     method=sn.TrustRegion(initial_trust_region_linf=1),
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
#         timeout_in_seconds=500,
#         ping_interval_in_seconds=50,
#         delete_disposable_files="all",
#     )
    

