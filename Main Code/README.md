# walking-droplet-diffraction
Python code for tracking and analysing walking oil droplet trajectories through single- and double-slit geometries (BSc Physics project, University of Bristol)

## Scripts

- **Tracking_Algorithm.py** — Real-time droplet detection and tracking using an Allied Vision camera (Vimba SDK). Outputs trajectory CSVs with frame, x, y coordinates.
- **Trajectory_Reader_-_Single_slit.py** — Post-processing and analysis of single-slit trajectory data: speed filtering, deflection angle measurement, Fraunhofer/Fresnel theory overlays.
- **Trajectory_Reader_-_Double_slit.py** — Post-processing and analysis of double-slit trajectory data: speed filtering, deflection angle histograms, double-slit interference theory overlays.

## Setup

**Note:** File paths in each script are hardcoded to local directories and will need to be updated before use. Edit the `FILEPATH` and `OUTPUT_FOLDER` variables at the top of each script to point to your own data and output locations.

### Dependencies

- Python 3
- NumPy, Pandas, Matplotlib, SciPy, OpenCV (`cv2`)
- `vmbpy` (Allied Vision Vimba SDK — required for `Tracking_Algorithm.py` only)

## Authors

Jacob Brooke and Harry Tonge
