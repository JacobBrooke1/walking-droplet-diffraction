# Walking Droplet Diffraction Experiment

This project investigates quantum-like behaviour in a classical hydrodynamic system, where millimetre-sized oil droplets "walk" across a vertically vibrated fluid bath and are guided by the waves they generate. The experiment explores phenomena analogous to quantum diffraction and interference using single- and double-slit geometries.

---

## Repository Structure

### `cad/`
CAD designs for the experimental apparatus, including:
- Slit module (single and double slit configurations)  
- Droplet launcher  
- LED mounting system  
- Double-slit centre piece  

Each component is provided in multiple formats (`.dwg`, `.ipt`, `.step`) to support reproducibility and fabrication.

---

### `code/`
Python and Arduino code used for data acquisition and analysis:
- Droplet tracking using computer vision (OpenCV)  
- Trajectory analysis for single- and double-slit experiments  
- Arduino control for automated droplet generation  

---

### `data/` *(if applicable)*  
Processed or example trajectory datasets used for analysis and validation.

---

## Summary

Together, the CAD designs and code provide a complete experimental and computational framework for studying pilot-wave hydrodynamics and its connection to quantum-like behaviour.
