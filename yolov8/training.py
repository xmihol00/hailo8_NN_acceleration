from ultralytics import YOLO
import torch.quantization as quant
import torch

model = YOLO("models/yolov8m.pt").to("cpu")                                        # load a pretrained model
model.qconfig = quant.get_default_qat_qconfig('fbgemm')                            # set the quantization configuration
model = quant.prepare_qat(model, inplace=True)                                     # prepare the model for quantization
results = model.train(data="datasets/plates/data.yaml", epochs=0, int8=True)       # train the model
model = quant.convert(model, inplace=True)                                         # convert the model to a quantized model
metrics = model.val()                                                              # evaluate model performance on the validation set
torch.jit.save(torch.jit.script(model), 'quantized_model.pth')                     # save the quantized model
