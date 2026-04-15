# from ultralytics import YOLO
# model = YOLO("data/activation_best.pt")
# print(model.names)

from ultralytics import YOLO

model = YOLO("data/activation_new.pt")

results = model(
    "test.jpg",
    conf=0.3
)

for r in results:
    for box in r.boxes:
        print(
            int(box.cls),
            model.names[int(box.cls)],
            float(box.conf)
        )



