import os
import shutil
import numpy as np
import cv2
from skimage import color
from tqdm import tqdm as tq
original_outer = "/Users/prad7599/Documents/TesterCodes/TryData/Face_shift_L+0.00_a+0.00_b+0.00_dE00.00"
face_albedo_rel = "Textures/JPG/Face/Face_Albedo.jpg"
output_base = "/Users/prad7599/Documents/TesterCodes/TryData"
os.makedirs(output_base, exist_ok=True)

albedo_path = os.path.join(original_outer, face_albedo_rel)
albedo_rgb = cv2.cvtColor(cv2.imread(albedo_path), cv2.COLOR_BGR2RGB) / 255.0  # normalize
albedo_lab = color.rgb2lab(albedo_rgb)
mean_lab = albedo_lab.reshape(-1,3).mean(axis=0)
print("Original mean Lab:", mean_lab)


np.random.seed(40)  

fine_bins = np.linspace(-5, 5, 5)  # 4 intervals
fine = np.array([np.random.uniform(fine_bins[i], fine_bins[i+1]) for i in range(4)])

coarse_bins_left = np.linspace(-20, fine.min(), 2)   # 1 interval
coarse_bins_right = np.linspace(fine.max(), 20, 3)   # 2 intervals
coarse = np.array([np.random.uniform(coarse_bins_left[i], coarse_bins_left[i+1]) 
                   for i in range(1)] +
                  [np.random.uniform(coarse_bins_right[i], coarse_bins_right[i+1]) 
                   for i in range(2)])

shift_values = np.unique(np.round(np.concatenate([fine, coarse]), 4))

print("Fine:", fine)
print("Coarse:", coarse)
print("All shift values:", shift_values)

dataset_count = 0
print("L shifts")
for L_shift in tq(shift_values):
    print("A shifts")
    for a_shift in tq(shift_values):
        print("B shifts")
        for b_shift in tq(shift_values):
            dataset_count += 1

            # Apply shifts
            delta = np.array([L_shift, a_shift, b_shift])
            albedo_lab_shifted = albedo_lab + delta

            # Clip Lab ranges
            albedo_lab_shifted[...,0] = np.clip(albedo_lab_shifted[...,0], 0, 100)
            albedo_lab_shifted[...,1] = np.clip(albedo_lab_shifted[...,1], -128, 127)
            albedo_lab_shifted[...,2] = np.clip(albedo_lab_shifted[...,2], -128, 127)

            # Compute delta E of mean color
            shifted_mean = albedo_lab_shifted.reshape(-1,3).mean(axis=0)
            deltaE = np.linalg.norm(shifted_mean - mean_lab)
            
            folder_name = f"Face_shift_L{L_shift:+.2f}_a{a_shift:+.2f}_b{b_shift:+.2f}_dE{deltaE:.2f}"
            folder_path = os.path.join(output_base, folder_name)

            # Copy the entire original folder
            if not os.path.exists(folder_path):
                shutil.copytree(original_outer, folder_path)

            # Convert back to RGB
            rgb_shifted = color.lab2rgb(albedo_lab_shifted)
            rgb8 = np.clip((rgb_shifted*255.0),0,255).astype(np.uint8)

            # Save shifted Face_Albedo.jpg
            save_path = os.path.join(folder_path, face_albedo_rel)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))

            # print(f"Saved dataset {dataset_count}: {folder_name} (ΔE={deltaE:.2f})")

print(f"\n Generated {dataset_count} shifted face datasets.")