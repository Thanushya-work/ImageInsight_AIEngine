from ultralytics import YOLO
from collections import defaultdict
import os
import csv
import cv2

# Load model
model = YOLO("data/capsmodel_22nd.pt")

# Input folder
image_folder = "caps_test"

# Output CSV
output_csv = "output_counts.csv"

# Output folder for annotated images
output_img_folder = "annotated_images"
os.makedirs(output_img_folder, exist_ok=True)

# Supported image formats
valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

rows = []

# Loop through all images
for img_name in os.listdir(image_folder):
    if not img_name.lower().endswith(valid_ext):
        continue

    img_path = os.path.join(image_folder, img_name)

    # Run inference
    results = model(
        img_path,
        conf=0.2,
        iou=0.5   # 🔥 FIXED (important)
    )

    class_counts = defaultdict(int)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            class_counts[class_name] += 1

        # ✅ Generate annotated image
        annotated_img = r.plot()

        # ✅ Save annotated image
        save_path = os.path.join(output_img_folder, img_name)
        cv2.imwrite(save_path, annotated_img)

    # Save results per class
    if class_counts:
        for cls, count in class_counts.items():
            rows.append([img_name, cls, count])
    else:
        rows.append([img_name, "No_Detection", 0])

# Write to CSV
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "class_name", "ai_count"])
    writer.writerows(rows)

print(f"CSV saved at: {output_csv}")
print(f"Annotated images saved in: {output_img_folder}")