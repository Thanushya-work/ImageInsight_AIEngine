# # from ultralytics import YOLO
# # model = YOLO("data/activation_april.pt")
# # print(model.names)  

# # # from ultralytics import YOLO

# # # model1 = YOLO("data/shelf_new_model.pt")
# # # model2 = YOLO("data/availability_may_5th.pt")

# # # print(model1.names == model2.names)

# # # from ultralytics import YOLO
# # # from collections import defaultdict
# # # import cv2

# # # model = YOLO("data/capsmodel_22nd.pt")

# # # results = model(
# # #     "a.jpg",
# # #     conf=0.2,iou=0.1
# # # )

# # # class_counts = defaultdict(int)

# # # for r in results:
# # #     # Count classes
# # #     for box in r.boxes:
# # #         class_id = int(box.cls)
# # #         class_name = model.names[class_id]
# # #         class_counts[class_name] += 1

# # #     # Get annotated image
# # #     annotated_img = r.plot()   # draws boxes + labels

# # #     # Save annotated image
# # #     cv2.imwrite("annotated_output.png", annotated_img)

# # # # Print class-wise counts
# # # print("Class-wise counts:")
# # # for cls, count in class_counts.items():
# # #     print(f"{cls}: {count}")

# # # print("Total detections:", sum(class_counts.values()))



from ultralytics import YOLO
from collections import defaultdict
import cv2
import os
import pandas as pd

model = YOLO("data/activation_april.pt")

image_paths = [
    "image (9).jpg",
    "image (10).jpg",
    "image (11).jpg",
    "image (13).jpg",
    "image (14).jpg",
    "image (15).jpg",
    "image (16).jpg",
    "image (17).jpg",
    "image (18).jpg"
]

results = model(image_paths, conf=0.01, iou=0.1)

overall_counts = defaultdict(int)
rows = []  # for CSV

for idx, r in enumerate(results):
    image_name = os.path.basename(image_paths[idx])
    class_counts = defaultdict(int)

    # Count SKUs
    for box in r.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        class_counts[class_name] += 1
        overall_counts[class_name] += 1

    # Save annotated image
    annotated_img = r.plot()
    cv2.imwrite(f"annotated_{image_name}", annotated_img)

    # Print per image
    print(f"\nImage: {image_name}")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

        # Save row for CSV
        rows.append({
            "image_name": image_name,
            "sku": cls,
            "count": count
        })

    print("Total:", sum(class_counts.values()))

# Save CSV
df = pd.DataFrame(rows)
df.to_csv("sku_counts_per_image.csv", index=False)

# Overall summary
print("\n=== Overall Counts ===")
for cls, count in overall_counts.items():
    print(f"{cls}: {count}")

print("Total detections:", sum(overall_counts.values()))


# from ultralytics import YOLO
# from collections import defaultdict
# import cv2
# import os
# import glob
# import pandas as pd

# model = YOLO("data/activation_april.pt")

# # Folder containing images
# image_folder = "activation_test"

# # Folder to save annotated images
# output_folder = "annotated_images"
# os.makedirs(output_folder, exist_ok=True)

# # Read all image files from folder
# image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
#               glob.glob(os.path.join(image_folder, "*.png")) + \
#               glob.glob(os.path.join(image_folder, "*.jpeg"))

# results = model(image_paths, conf=0.01, iou=0.1)

# overall_counts = defaultdict(int)
# rows = []  # for CSV

# for idx, r in enumerate(results):
#     image_name = os.path.basename(image_paths[idx])
#     class_counts = defaultdict(int)

#     # Count SKUs
#     for box in r.boxes:
#         class_id = int(box.cls)
#         class_name = model.names[class_id]
#         class_counts[class_name] += 1
#         overall_counts[class_name] += 1

#     # Save annotated image
#     annotated_img = r.plot()

#     annotated_image_path = os.path.join(
#         output_folder,
#         f"annotated_{image_name}"
#     )

#     cv2.imwrite(annotated_image_path, annotated_img)

#     # Print per image
#     print(f"\nImage: {image_name}")
#     for cls, count in class_counts.items():
#         print(f"{cls}: {count}")

#         # Save row for CSV
#         rows.append({
#             "image_name": image_name,
#             "sku": cls,
#             "count": count
#         })

#     print("Total:", sum(class_counts.values()))

# # Save CSV
# df = pd.DataFrame(rows)
# df.to_csv("sku_counts_per_image.csv", index=False)

# # Overall summary
# print("\n=== Overall Counts ===")
# for cls, count in overall_counts.items():
#     print(f"{cls}: {count}")

# print("Total detections:", sum(overall_counts.values()))