import trimesh
import os
from dataclasses import dataclass

@dataclass
class MeshScaler:
    scale_factor: float
    file_dir: str = "/Users/blambright/Downloads/Capstone/main_code/e-NABLE_Phoenix_Hand_v3-4056253/files"
    init_output_dir: str = "/Users/blambright/Downloads/Capstone/main_code/e-NABLE_Phoenix_Hand_v3-4056253-modified/"

    def load_stl_files(self):
        dirs = os.listdir(self.file_dir)
        stl_paths = [os.path.join(self.file_dir, f) for f in dirs if f.endswith('.stl')]
        return stl_paths

    def modify_mesh(self, mesh_path: str, output_path: str):
        # Load the mesh
        mesh = trimesh.load(mesh_path)
        # Apply scaling
        mesh.apply_scale(self.scale_factor)
        # Save the modified mesh
        mesh.export(output_path)
        print(f"Modified mesh saved as {output_path}")

    def process_all_meshes(self):
        stl_paths = self.load_stl_files()
        os.makedirs(self.init_output_dir, exist_ok=True)

        output_paths = []
        for stl_path in stl_paths:
            base, ext = os.path.splitext(os.path.basename(stl_path))
            output_path = f"{base}_scaled{ext}"
            new_stl_path = os.path.join(self.init_output_dir, output_path)
            output_paths.append(new_stl_path)
            self.modify_mesh(stl_path, new_stl_path)
        
        return output_paths
    
def calculate_scale(length: float, width: float) -> float:
    """Calculate scale factor based on the largest dimension."""
    original_length = 130
    original_width = 60
    scales = [100 + x for x in range(0, 65, 5)]  # 100 to 160 mm
    scales = [s / 100 for s in scales]  # convert to scale factors

    fits = []
    for scale in scales:
        scaled_length = (original_length * scale) - length
        scaled_width = (original_width * scale) - width
        fit_quality = 100 - abs((scaled_length + scaled_width) / 2)
        fits.append((fit_quality, scale))

    fits.sort(reverse=True)
    best_fit = fits[0]
    scale_factor = best_fit[1]

    return scale_factor

# print(calculate_scale(120, 140))

# load stl files
# stl_dir = "/Users/blambright/Downloads/Capstone/main_code/e-NABLE_Phoenix_Hand_v3-4056253-2/files"
# dirs = os.listdir(stl_dir)
# stl_paths = [os.path.join(stl_dir, f) for f in dirs if f.endswith('.stl')]

# # now scale them according to some factor
# scale_factor = 1.25

# # scale each mesh and save
# for stl_path in stl_paths:
#     mesh = trimesh.load(stl_path)
#     mesh.apply_scale(scale_factor)

#     # === Save modified mesh ===
#     base, ext = os.path.splitext(stl_path)
#     new_stl_path = f"{base}_scaled{ext}"
#     mesh.export(new_stl_path)

#     print(f"Modified mesh saved as {new_stl_path}")


# example on a single document
# example_stl = "/Users/blambright/Downloads/Capstone/main_code/e-NABLE_Phoenix_Hand_v3-4056253/files/Arm_Guard.stl"
# mesh = trimesh.load(example_stl)
# mesh.apply_scale(scale_factor)
# # save modified mesh
# base, ext = os.path.splitext(example_stl)
# new_stl_path = f"{base}_scaled{ext}"
# mesh.export(new_stl_path)