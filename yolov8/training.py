from ultralytics import YOLO

model = YOLO("models/yolov8m.pt")                        # load a pretrained model
model.train(data="datasets/plates/data.yaml", epochs=3)  # train the model
metrics = model.val()                                    # evaluate model performance on the validation set
