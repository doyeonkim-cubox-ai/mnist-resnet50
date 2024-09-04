import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import argparse


def main():
    # Add parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='Image path(ex. /**/image.*)')
    img_path = parser.parse_args()

    image = Image.open(img_path.img)
    image = image.resize((28, 28)).convert('L')
    img_tensor = transforms.ToTensor()(image)
    img_tensor = img_tensor.unsqueeze(0)

    session = ort.InferenceSession("./model/model.onnx", providers=['CPUExecutionProvider'])

    ort_in = {session.get_inputs()[0].name: np.array(img_tensor)}
    ort_out = session.run(None, ort_in)[0]

    # postprocess
    ort_out = ort_out.squeeze(0)
    res = np.argmax(ort_out)
    print(res)


if __name__ == "__main__":
    main()