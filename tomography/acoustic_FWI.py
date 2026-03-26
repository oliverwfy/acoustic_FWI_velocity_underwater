import salvus.namespace as sn
import numpy as np
from my_code.utilities import *
from pathlib import Path
import salvus
import salvus.mesh.layered_meshing as lm
from datetime import datetime
import math
import xarray as xr

# # Directories in WSL
# PROJECT_DIR = '/home/oliver/workspace/Salvus/acoustic_model/Project'
# IMAGE_DIR = '/home/oliver/workspace/Salvus/acoustic_model/image'
# DATA_DIR = '/home/oliver/workspace/Salvus/acoustic_model/data'


# Directories in Windows
PROJECT_DIR_WIN = '/home/b6as/oliverwfy.b6as/workspace/acoustic_model/Project'
DATA_DIR_WIN = '/home/b6as/oliverwfy.b6as/workspace/acoustic_model/data'
IMAGE_DIR_WIN = '/home/b6as/oliverwfy.b6as/workspace/acoustic_model/image'


# create dir if it does not exist

Path(PROJECT_DIR_WIN).mkdir(parents=True, exist_ok=True)
Path(IMAGE_DIR_WIN).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR_WIN).mkdir(parents=True, exist_ok=True)



# salvus site 
SITE_NAME = "isambard_oliver"
RANKS = 8


CENTRAL_FREQUENCY = 2e4  # Hz
f_c = 2e4  # central frequency in Hz
PROJECT_NAME = 'acoustic_forward_5paris'


VP = 5000.0
RHO = 2600.0


x0, x1 = 0.0, 1
y0, y1 = 0.0, 1


domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)

p = sn.Project.from_domain(
    path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True
)



# # Define layers
# layer_ls = [{ 'rho': RHO, 'vp': VP , 'd':1.0}, { 'rho': RHO, 'vp': VP*1.1 , 'd':2.0}, { 'rho': RHO, 'vp': VP*1.3 , 'd':2.0}, { 'rho': RHO, 'vp': VP*1.5 , 'd':2.0}]

# hyperplane_ls = np.cumsum([layer['d'] for layer in layer_ls][::-1])


# layered_model_ls = []

# for i in range(len(layer_ls)):
#     layered_model_ls.append(sn.material.acoustic.Velocity.from_params(rho=layer_ls[i]['rho'], vp=layer_ls[i]['vp']))
#     layered_model_ls.append(sn.layered_meshing.interface.Hyperplane.at(hyperplane_ls[i]))

# layered_model_ls.pop() # delete the last hyperplane


# layered_model = lm.LayeredModel(layered_model_ls)


vp_grad = xr.DataArray(
    np.linspace(1.0, 1.5, 6), [("y", np.linspace(1.0, 0.0, 6))]
)


m1 = sn.material.from_params(rho=RHO, vp=VP*vp_grad)



# mesh resolution parameters
elements_per_wavelength = 3  # number of elements per wavelength
model_order = 4  # model order for the mesh


# absorbing boundary parameters
reference_velocity = 5000           # wave velocity in the absorbing boundary layer
number_of_wavelengths=2            # number of wavelengths to pad the domain by
reference_frequency = f_c*2           # reference frequency for the distance calculation
free_surfaces = ['y0', 'y1']       # free surfaces, absorbing boundaries are applied for the rest






m_homo = sn.material.from_params(rho=RHO, vp=VP)

# adding absorbing boundary conditions to the layered model
model_ab_homo = sn.layered_meshing.MeshingProtocol(
    m_homo,
    ab=salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
        free_surface=free_surfaces,
        number_of_wavelengths=number_of_wavelengths,
        reference_velocity=reference_velocity,
        reference_frequency=reference_frequency,
    ),
)

# create mesh 
mesh_homo = lm.mesh_from_domain(
    domain=domain,
    model=model_ab_homo,
    mesh_resolution=sn.MeshResolution(
        reference_frequency=reference_frequency, elements_per_wavelength=elements_per_wavelength, model_order=model_order
    ),
)





# adding absorbing boundary conditions to the layered model
model_ab = sn.layered_meshing.MeshingProtocol(
    m1,
    ab=salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
        free_surface=free_surfaces,
        number_of_wavelengths=number_of_wavelengths,
        reference_velocity=reference_velocity,
        reference_frequency=reference_frequency,
    ),
)


# create mesh 
mesh = lm.mesh_from_domain(
    domain=domain,
    model=model_ab,
    mesh_resolution=sn.MeshResolution(
        reference_frequency=reference_frequency, elements_per_wavelength=elements_per_wavelength, model_order=model_order
    ),
)




aplitude_ratio=1e5

s_loc = (0.5, 0.2)
r_loc = [(i, 1.0) for i in np.linspace(0.2, 0.8, 5)]

# receiver = sn.simple_config.receiver.cartesian.Point2D(x=r_loc[0], y=r_loc[1], station_code="R1", fields=["phi"])   

receivers = [sn.simple_config.receiver.cartesian.Point2D(x=i[0], y=i[1], station_code=f"R{j+1}", fields=["phi"]) for j, i in enumerate(r_loc)]

source = sn.simple_config.source.cartesian.ScalarPoint2D(x=s_loc[0], y=s_loc[1], f=aplitude_ratio)


p.add_to_project(
    sn.Event(event_name="event_1", sources=source, receivers=receivers)
)



"""
Configuration from parameters

"""

end_time = 5e-4
start_time= 0
sampling_rate_in_hertz = CENTRAL_FREQUENCY* 200 

time_shift_in_seconds=1/f_c
# simulation of homogeneous model
simulation_name = 'forward_simulation_homogeneous_model'

wavelet = sn.simple_config.stf.Ricker(center_frequency=f_c, time_shift_in_seconds=time_shift_in_seconds)     # wavelet (input source time function) 
# wavelet = sn.simple_config.stf.Ricker(center_frequency=f_c)     # wavelet (input source time function) 



# waveform simulation configuration
waveform_config = sn.WaveformSimulationConfiguration(
        start_time_in_seconds=start_time,
        end_time_in_seconds=end_time,
        time_step_in_seconds=1/sampling_rate_in_hertz
        )

# event configuaration
event_config = sn.EventConfiguration(
    wavelet=wavelet,
    waveform_simulation_configuration=waveform_config,
    )



sim_config = sn.UnstructuredMeshSimulationConfiguration(
    unstructured_mesh=mesh_homo,
    name=simulation_name,
    event_configuration=event_config,
    )



# add simulation configuration to Project
p.add_to_project(
    sim_config, overwrite=True
    )




# simulation of layered model 
simulation_name = 'forward_simulation_layred_model'


# waveform simulation configuration
waveform_config = sn.WaveformSimulationConfiguration(
        start_time_in_seconds=start_time,
        end_time_in_seconds=end_time,
            time_step_in_seconds=1/sampling_rate_in_hertz

        )

# event configuaration
event_config = sn.EventConfiguration(
    wavelet=wavelet,
    waveform_simulation_configuration=waveform_config,
    )



sim_config = sn.UnstructuredMeshSimulationConfiguration(
    unstructured_mesh=mesh,
    name=simulation_name,
    event_configuration=event_config,
    )



# add simulation configuration to Project
p.add_to_project(
    sim_config, overwrite=True
    )


# # visualization of mesh and simulation set-up
# p.viz.nb.simulation_setup(
#     simulation_configuration=simulation_name, events=p.events.list()
#     )




# simulation_name = 'forward_simulation_homogeneous_model'
simulation_name = 'forward_simulation_layred_model'


dofs =  mesh.number_of_nodes
print(f'Start simulation: {simulation_name}')
print(f'Dofs (number of nodes): {dofs}')


"""
Launch simulations

"""

start_time = datetime.now()


# p.simulations.launch(
#     ranks_per_job=RANKS,
#     site_name=SITE_NAME,
#     events=p.events.list(),
#     simulation_configuration=simulation_name,
#     delete_conflicting_previous_results=True,
#     )


# simulation with volume data (full wavefield)
p.simulations.launch(
    ranks_per_job=RANKS,
    site_name=SITE_NAME,
    events=p.events.list(),
    simulation_configuration=simulation_name,
    extra_output_configuration={
        "volume_data": {
            "sampling_interval_in_time_steps": 10,
            "fields": ["phi", "gradient-of-phi"],
        },
    },
    # We have previously simulated the same event but without
    # extra output. We have to thus overwrite the existing
    # simulation.
    delete_conflicting_previous_results=True,
)



p.simulations.query(block=True)


end_time = datetime.now()


execution_time_seconds = (end_time - start_time).total_seconds()
minutes = int(execution_time_seconds // 60)  # Extract minutes
seconds = execution_time_seconds % 60  # Extract remaining seconds

print(f"Execution time: {minutes} minutes and {seconds:.2f} seconds")











# # The misfit configuration defines how synthetics are compared to observed data.
# p += sn.MisfitConfiguration(
#     name="L2_misfit",
#     # Could be observed data. Here we compare to the synthetic target.
#     observed_data="forward_simulation_layred_model",
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
#     prior_model="forward_simulation_homogeneous_model",
#     # The events to use.
#     events=event_names_imaging,
#     misfit_configuration="L2_misfit",
#     # What parameters to invert for.
#     mapping=sn.Mapping(
#         scaling="relative_deviation_from_prior",
#         inversion_parameters=["VP"],
#         # region_of_interest=mesh_roi,
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
    