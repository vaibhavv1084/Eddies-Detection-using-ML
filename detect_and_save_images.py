import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Import your model builder from FrameWork.py
from FrameWork import Deeplabv3, BilinearUpsampling, SepConv_BN, _conv2d_same, \
                      _xception_block, relu6, _make_divisible, _inverted_res_block

# 2. Paths
WEIGHTS_PATH = './eddydlv3net/eddynet.weights.h5'
SSH_NPY_PATH = '../../Sea/filtered_SSH_train_data.npy'   # or filtered_SSH_test_data.npy
OUTPUT_DIR = './detected_eddies'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Load the SSH anomaly data
#    Shape: (num_days, H, W)
ssh_data = np.load(SSH_NPY_PATH)

# 4. Build the model (must match exactly what you trained)
#    Adjust input_shape to match your data (e.g. (H, W, 1))
num_days, H, W = ssh_data.shape
model = Deeplabv3(input_shape=(H, W, 1), classes=3)
model.load_weights(WEIGHTS_PATH)
print(f"Model loaded with weights from {WEIGHTS_PATH}")

# 5. Loop over each day, predict and save overlay
for idx in range(num_days):
    img = ssh_data[idx]
    x = img[np.newaxis, ..., np.newaxis]      # shape (1, H, W, 1)
    
    # model.predict → shape (1, H, W, 3)
    pred = model.predict(x)[0]
    mask = np.argmax(pred, axis=-1)           # shape (H, W), values in {0,1,2}
    
    # Plot the anomaly map
    plt.figure(figsize=(6,5))
    plt.imshow(img, cmap='jet', clim=(-0.3, 0.3))
    
    # Overlay eddy masks as contours
    # Here: contour class==1 (anti-cyclonic) in red, class==2 (cyclonic) in blue
    levels = [0.5, 1.5, 2.5]
    contour_colors = ['red', 'blue']
    for c, color in zip([1,2], contour_colors):
        plt.contour(mask==c, levels=[0.5], colors=[color], linewidths=1)
    
    plt.axis('off')
    plt.title(f"Day {idx:04d}")
    
    # Save figure
    out_path = os.path.join(OUTPUT_DIR, f"eddy_day_{idx:04d}.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    if idx % 50 == 0:
        print(f"  → Saved {out_path}")

print(f"\n✅ All {num_days} days processed. Check `{OUTPUT_DIR}` for your eddy-overlay images.")
