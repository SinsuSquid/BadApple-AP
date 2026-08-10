# 🍎 [Bad Apple!! but it's on Multi-Scale Apollonian Particle Simulation](https://youtu.be/6OdOjmtIa1g)

![BadApple-AP](bad_apple_5sec.gif)

A computational physics and materials science visual flex that renders the iconic [Bad Apple!!](https://www.youtube.com/watch?v=FtutLA63Cp8&list=RDFtutLA63Cp8&start_radio=1&pp=ygUJYmFkIGFwcGxloAcB) shadow art PV using 3D Euclidean Distance Transforms (EDT) and Multi-Scale Apollonian Sphere Packing exported as a parallelized LAMMPS trajectory (`.lammpstrj`) for real-time 30 FPS playback in OVITO.

## 🔬 How It Works
Instead of rendering a uniform 2D pixel grid or fixed-size 3D voxel lattice, this pipeline transforms each video frame into a multi-scale atomic system:
1. **3D Volumetric Extrusion**: Each video frame is inverted (black shadow = active volume) and extruded along the $Z$-axis using a Gaussian thickness profile $e^{-z^2}$.
2. **3D Euclidean Distance Field**: `scipy.ndimage.distance_transform_edt` calculates the exact distance from every interior voxel to the nearest boundary.
3. **Apollonian Sphere Packing**: High-distance core regions are stuffed with large macro-particles ($\text{Au}, \text{Fe}$), while smaller micro-particles ($\text{Si}, \text{C}$) dynamically fill remaining boundary gaps and fine features.
4. **LAMMPS Dump Generation**: Each frame is formatted with custom `ITEM: ATOMS` headers containing atomic coordinates, species type, charge $q$, and calculated radii.
5. **Parallel Execution**: Leverages Python's `concurrent.futures.ProcessPoolExecutor` to distribute frame rendering across all available logical CPU cores.

## 🛠️ Requirements & Installation
- Python Environment:
  - Python 3.9+
  - opencv-python
  - numpy
  - scipy
- Install dependencies via pip or conda:
  ```
  pip install opencv-python numpy scipy
  ```
- Visualizer: [OVITO (Open Visualization Tool)](https://www.ovito.org/) (Basic or Pro)

## 🚀 Usage
1. Run Parallel Trajectory Generator
   Place `BadApple.mp4` in the root project folder and execute the OpenMP parallel script
   ```
   python bad_apple_openmp.py
   ```
   The script will automatically detect your CPU core count and output `bad_apple_multiscale.lammpstrj`.
2. Visualize in OVITO
   1. Drag and drop `bad_apple_multiscale.lammpstrj` into OVITO.
   2. Click Animation Settings (⚙️) at the bottom right of the timeline and set FPS to 30.0.
   3. Add Modifier: Color Coding ➔ Set property to Radius or Type and select your favorite color palette (e.g., Plasma, Hot, or Spectral).(Optional)
   4. Add Modifier: Construct Surface Mesh to render an isosurface wrapping around the packed particle cloud.
   5. Hit Play ▶️!

## 📜 LicenseMIT License.
Free to adapt for other media or simulation research flexes! (๑˃̵ᴗ˂̵)ﻭ 🚀💖
