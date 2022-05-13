import numpy as np
import cv2
import torch.onnx
from stylenet import StyleNetwork
from torchvision import transforms as T

if __name__ == '__main__':

    preprocess = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    net = StyleNetwork('./models/style_7.pth')

    for p in net.parameters():
        p.requires_grad = False

    net = net.eval()

    # NCHW
    dummy_input = torch.randn(1, 3, 720, 1280, requires_grad=True)

    # Export the model
    torch.onnx.export(net,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "models/StyleTransfer_7.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput']#,  # the model's output names
                      #dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                      #              'modelOutput': {0: 'batch_size'}}
                      )
