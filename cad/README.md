# CAD Designs – Walking Droplet Diffraction Experiment

This directory contains the CAD files used to construct the experimental apparatus for the walking droplet diffraction project.

## Overview

The components in this folder were designed to enable controlled and repeatable experiments investigating hydrodynamic quantum analogues, including single-slit diffraction and double-slit interference.

The designs prioritise stability, modularity, and minimal disturbance to the fluid system, particularly in the high-memory regime near the Faraday threshold.

---

## Components

### Slit Module (`slits.ipt`)
- Primary experimental insert used for both single- and double-slit configurations  
- Enables adjustable slit geometries for diffraction experiments  
- Designed to sit as a submerged barrier within the vibrating bath  

### Droplet Launcher (`droplet_launcher_v2.ipt`)
- Final design used for controlled droplet generation  
- Compatible with an Arduino-controlled relay system  
- Based on an improved design inspired by prior experimental work (e.g. Pucci et al., 2017)  

### LED Mount (`led_mount.ipt`)
- Mounting system for illumination of the droplet  
- Ensures consistent lighting conditions for high-contrast imaging and tracking  

### Double Slit Centre Piece (`double_slit_centre_piece_5mm.ipt`)
- Central insert for double-slit experiments  
- Tested as part of slit geometry development  
- Included for completeness and potential reuse  

---

## Design Philosophy

All components were designed to:

- Ensure reproducible experimental conditions  
- Minimise vibrations and external disturbances  
- Allow modular swapping of experimental geometries  
- Maintain precise alignment within the bath  

---

## Notes

- All files are provided in Autodesk Inventor (`.ipt`) format  
- Designs can be exported to `.STL` or `.STEP` for fabrication if required  
- The final experimental configuration primarily used the slit module and droplet launcher  

---

## Reproducibility

Together with the analysis code in the main repository, these CAD designs provide a complete description of the experimental system, enabling replication of the walking droplet diffraction experiments.
