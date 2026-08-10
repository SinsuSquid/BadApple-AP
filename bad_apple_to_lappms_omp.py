import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import sys

def process_single_frame(args):
    """Worker task for parallel execution."""
    frame_idx, frame_gray, grid_w, grid_h, grid_z, radii_tiers = args

    # 1. Image Preprocessing & Inversion
    resized = cv2.resize(frame_gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    density_2d = 1.0 - (resized / 255.0)  # Shadow = 1.0

    # 2. Gaussian Z-Extrusion
    z_indices = np.linspace(-2.0, 2.0, grid_z)
    z_gaussian = np.exp(-z_indices**2)

    vol_3d = np.zeros((grid_w, grid_h, grid_z), dtype=bool)
    for z in range(grid_z):
        vol_3d[:, :, z] = (density_2d.T * z_gaussian[z]) > 0.35

    # 3. 3D Distance Transform (C-accelerated OpenMP/SciPy)
    dist_map = distance_transform_edt(vol_3d)

    # 4. Multi-Scale Apollonian Packing
    packed_particles = []
    occupied = np.zeros_like(vol_3d, dtype=bool)
    atom_id = 1
    elements = ["Au", "Fe", "Si", "C"]

    for tier_idx, r_target in enumerate(radii_tiers):
        atom_type = tier_idx + 1
        elem = elements[min(tier_idx, len(elements) - 1)]

        candidates = (dist_map >= r_target) & (~occupied)
        candidate_indices = np.argwhere(candidates)
        np.random.shuffle(candidate_indices)

        for cx, cy, cz in candidate_indices:
            if occupied[cx, cy, cz]:
                continue

            x_min, x_max = max(0, int(cx - r_target)), min(grid_w, int(cx + r_target + 1))
            y_min, y_max = max(0, int(cy - r_target)), min(grid_h, int(cy + r_target + 1))
            z_min, z_max = max(0, int(cz - r_target)), min(grid_z, int(cz + r_target + 1))

            X, Y, Z = np.ogrid[x_min-cx:x_max-cx, y_min-cy:y_max-cy, z_min-cz:z_max-cz]
            sphere_mask = (X**2 + Y**2 + Z**2) <= (r_target**2)

            if not np.any(occupied[x_min:x_max, y_min:y_max, z_min:z_max] & sphere_mask):
                occupied[x_min:x_max, y_min:y_max, z_min:z_max] |= sphere_mask

                px = cx * 0.5
                py = (grid_h - cy) * 0.5  # Flip Y
                pz = cz * 0.5
                radius_angstroms = r_target * 0.25

                packed_particles.append((
                    atom_id, atom_type, elem, px, py, pz, 0.0, radius_angstroms
                ))
                atom_id += 1

    box_x = grid_w * 0.5
    box_y = grid_h * 0.5
    box_z = grid_z * 0.5

    # Format frame block as string
    lines = [
        "ITEM: TIMESTEP",
        f"{frame_idx}",
        "ITEM: NUMBER OF ATOMS",
        f"{len(packed_particles)}",
        "ITEM: BOX BOUNDS pp pp pp",
        f"0.000000 {box_x:.6f}",
        f"0.000000 {box_y:.6f}",
        f"0.000000 {box_z:.6f}",
        "ITEM: ATOMS id type element x y z q radius"
    ]
    for p in packed_particles:
        lines.append(f"{p[0]} {p[1]} {p[2]} {p[3]:.3f} {p[4]:.3f} {p[5]:.3f} {p[6]:.2f} {p[7]:.3f}")

    return frame_idx, "\n".join(lines) + "\n"


def main():
    video_path = "BadApple.mp4"
    output_path = "bad_apple_multiscale.lammpstrj"
    grid_w, grid_h, grid_z = 120, 90, 20
    radii_tiers = [3.5, 2.0, 1.0, 0.5]

    num_workers = os.cpu_count()
    print(f"[OpenMP / ProcessPool] Launching with {num_workers} CPU worker threads/processes...", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    frame_tasks = []
    frame_idx = 0

    print("[1/3] Reading and extracting video frames into memory...", flush=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        task_args = (frame_idx, gray, grid_w, grid_h, grid_z, radii_tiers)
        frame_tasks.append(task_args)
        frame_idx += 1
    cap.release()

    total_frames = len(frame_tasks)
    print(f"[2/3] Total Frames Loaded: {total_frames}. Starting parallel rendering...", flush=True)

    start_time = time.time()
    results = {}

    # Parallel Execution across all CPU cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_frame, task): task[0] for task in frame_tasks}

        completed = 0
        for future in as_completed(futures):
            f_idx, frame_str = future.result()
            results[f_idx] = frame_str
            completed += 1

            if completed % 25 == 0 or completed == total_frames:
                elapsed = time.time() - start_time
                fps = completed / elapsed
                print(f" -> Progress: {completed}/{total_frames} frames completed ({fps:.2f} frames/sec)", flush=True)

    print("[3/3] Writing ordered trajectory file to disk...", flush=True)
    with open(output_path, "w") as f:
        for idx in range(total_frames):
            f.write(results[idx])

    total_elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Completed in {total_elapsed:.2f} seconds! Output saved to: {output_path}", flush=True)

if __name__ == "__main__":
    # Windows Multiprocessing Guard
    main()
