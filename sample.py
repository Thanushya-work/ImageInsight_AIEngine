# # from ultralytics import YOLO
# # model = YOLO("data/activation_best.pt")
# # print(model.names)

# from ultralytics import YOLO

# model = YOLO("data/weights.pt")

# results = model(
#     "test.jpg",
#     conf=0.12
# )

# for r in results:
#     for box in r.boxes:
#         print(
#             int(box.cls),
#             model.names[int(box.cls)],
#             float(box.conf)
#         )


import boto3
import json

# Load credentials
with open("aws_credentials.json") as f:
    creds = json.load(f)

s3 = boto3.client(
    's3',
    aws_access_key_id=creds["aws_access_key_id"],
    aws_secret_access_key=creds["aws_secret_access_key"],
    region_name=creds.get("region", "ap-south-1")
)

bucket_name = "imageinsightimagesbeverages"
prefix = "ModelResults/VisibleItem_303/"

count = 0
paginator = s3.get_paginator('list_objects_v2')

for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
            count += 1

print("Total images:", count)

