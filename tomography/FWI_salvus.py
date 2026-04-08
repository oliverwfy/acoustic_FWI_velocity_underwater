"""
Acoustic Full Waveform Inversion using Salvus
==============================================
2-D acoustic FWI on a unit square domain [0,1]^2.

Workflow:
  1. Build homogeneous starting mesh and layered true mesh.
  2. Define source–receiver geometry and simulation configuration.
  3. Run forward simulation on the true model to generate observed data.
  4. Configure L2 misfit and inversion problem.
  5. Iterate the inversion loop.
"""

# ============================================================
# Imports
# ============================================================
import sys
import importlib
from pathlib import Path
from datetime import datetime

import numpy as np
import salvus
import salvus.namespace as sn
import salvus.mesh.layered_meshing as lm

# Reload local utilities so edits take effect without restarting
path_to_add = str(Path.cwd() / "tomography")
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

import my_code.utilities
importlib.reload(my_code.utilities)
from my_code.utilities import *


# ============================================================
# Configuration
# ============================================================

# --- HPC site ---
SITE_NAME = "isambard_oliver"
RANKS     = 8

# --- Physical parameters ---
f_c  = 2e4    # centre frequency [Hz]
VP   = 5000.0 # P-wave velocity [m/s]
RHO  = 2600.0 # density [kg/m^3]

# --- Domain (unit square) ---
x0, x1 = 0.0, 1.0
y0, y1 = 0.0, 1.0

# --- Paths ---
PROJECT_DIR = '/home/b6as/oliverwfy.b6as/workspace/acoustic_model/Project'
DATA_DIR    = '/home/b6as/oliverwfy.b6as/workspace/acoustic_model/data'
IMAGE_DIR   = '/home/b6as/oliverwfy.b6as/workspace/acoustic_model/image'

for d in [PROJECT_DIR, DATA_DIR, IMAGE_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ============================================================
# Project
# ============================================================

PROJECT_NAME = 'acoustic_forward_5paris_salvus'
domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)

p = sn.Project.from_domain(
    path=Path(PROJECT_DIR, PROJECT_NAME), domain=domain, load_if_exists=True
)


# ============================================================
# Mesh construction
# ============================================================

# --- Resolution / absorbing-boundary settings ---
reference_frequency     = f_c * 2   # meshing reference frequency [Hz]
elements_per_wavelength = 3
model_order             = 4
number_of_wavelengths   = 2
free_surfaces           = ['y0', 'y1']

ab_params = salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
    free_surface=free_surfaces,
    number_of_wavelengths=number_of_wavelengths,
    reference_velocity=VP,
    reference_frequency=reference_frequency,
)

mesh_res = sn.MeshResolution(
    reference_frequency=reference_frequency,
    elements_per_wavelength=elements_per_wavelength,
    model_order=model_order,
)

# --- Homogeneous starting model (uniform VP, RHO) ---
m_homo   = sn.material.from_params(rho=RHO, vp=VP)
mesh_homo = lm.mesh_from_domain(
    domain=domain,
    model=sn.layered_meshing.MeshingProtocol(m_homo, ab=ab_params),
    mesh_resolution=mesh_res,
)

# --- True model: horizontal layer with VP *= 1.5 between y = 0.4 and y = 0.6 ---
mesh_true = mesh_homo.copy()
centroids  = mesh_true.get_element_centroid()
defect_mask = (centroids[:, 1] >= 0.4) & (centroids[:, 1] <= 0.6)
mesh_true.elemental_fields["VP"][defect_mask] = 1.5 * VP

# --- Region of interest for inversion: full x, y in [0.4, 0.6] ---
roi      = [(0.0, 1.0), (0.4, 0.6)]
mesh_roi = generate_mesh_roi(mesh_homo, roi)


# ============================================================
# Source–receiver geometry
# ============================================================

event_5p       = '5_pairs'
amplitude_ratio = 1e3

s_loc = (0.5, 0.3)                                      # single source
r_loc = [(x, 0.7) for x in np.linspace(0.2, 0.8, 5)]  # 5 receivers

source = sn.simple_config.source.cartesian.ScalarPoint2D(
    x=s_loc[0], y=s_loc[1], f=amplitude_ratio
)
receivers = [
    sn.simple_config.receiver.cartesian.Point2D(
        x=r[0], y=r[1], station_code=f"R{j+1}", fields=["phi"]
    )
    for j, r in enumerate(r_loc)
]

p.add_to_project(
    sn.Event(event_name=event_5p, sources=source, receivers=receivers)
)


# ============================================================
# Simulation configuration (wavelet + time axis)
# ============================================================

end_time               = 2e-4   # [s]
start_time             = 0.0    # [s]
sampling_rate_in_hertz = f_c * 100
time_shift_in_seconds  = 1.0 / f_c

wavelet = sn.simple_config.stf.Ricker(
    center_frequency=f_c,
    time_shift_in_seconds=time_shift_in_seconds,
)

waveform_config = sn.WaveformSimulationConfiguration(
    start_time_in_seconds=start_time,
    end_time_in_seconds=end_time,
    time_step_in_seconds=1.0 / sampling_rate_in_hertz,
)

event_config = sn.EventConfiguration(
    wavelet=wavelet,
    waveform_simulation_configuration=waveform_config,
)

# Register both simulation configurations with the project
homo_model = 'forward_simulation_homogeneous_model'
true_model = 'forward_simulation_layered_model'

p.add_to_project(
    sn.UnstructuredMeshSimulationConfiguration(
        unstructured_mesh=mesh_homo,
        name=homo_model,
        event_configuration=event_config,
    ),
    overwrite=True,
)
p.add_to_project(
    sn.UnstructuredMeshSimulationConfiguration(
        unstructured_mesh=mesh_true,
        name=true_model,
        event_configuration=event_config,
    ),
    overwrite=True,
)


# ============================================================
# Forward simulation — generate observed data from true model
# ============================================================

t0 = datetime.now()

forward_simulation(
    p,
    simulation_name=true_model,
    events=event_5p,
    fields=None,
    sampling_interval_in_time_steps=10,
    RANKS=RANKS,
)

print(f"Forward simulation completed in {datetime.now() - t0}")


# ============================================================
# Misfit configuration
# ============================================================

misfit_name = 'L2_misfit'

# Compare synthetics from the starting model against the true-model data
p += sn.MisfitConfiguration(
    name=misfit_name,
    observed_data=true_model,
    misfit_function="L2",
    receiver_field="phi",   # acoustic potential field
)


# ============================================================
# Inverse problem configuration
# ============================================================

inverse_config_name = 'inversion_L2'

invconfig = sn.InverseProblemConfiguration(
    name=inverse_config_name,
    prior_model=homo_model,
    events=event_5p,
    misfit_configuration=misfit_name,
    mapping=sn.Mapping(
        scaling="relative_deviation_from_prior",
        inversion_parameters=["VP"],
        region_of_interest=mesh_roi,
    ),
    method=sn.TrustRegion(initial_trust_region_linf=1.0),
    # Compress forward wavefield by subsampling in time to save memory
    wavefield_compression=sn.WavefieldCompression(
        forward_wavefield_sampling_interval=10
    ),
    job_submission=sn.SiteConfig(site_name=SITE_NAME, ranks_per_job=RANKS),
)

add_inversion(p, invconfig)

p.inversions.set_job_submission_configuration(
    inverse_config_name, sn.SiteConfig(site_name=SITE_NAME, ranks_per_job=RANKS)
)


# ============================================================
# Inversion loop
# ============================================================

max_iterations           = 10
timeout_in_seconds       = 360
ping_interval_in_seconds = 50
delete_disposable_files  = "all"

for i in range(max_iterations):
    p.inversions.iterate(
        inverse_problem_configuration=inverse_config_name,
        timeout_in_seconds=timeout_in_seconds,
        ping_interval_in_seconds=ping_interval_in_seconds,
        delete_disposable_files=delete_disposable_files,
    )
