from ultralytics import YOLO
import torch.quantization as quant

model = YOLO("models/yolov8m.pt")                        # load a pretrained model
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model = quant.prepare_qat(model)
model.train(data="datasets/plates/data.yaml", epochs=3)  # train the model
metrics = model.val()                                    # evaluate model performance on the validation set
