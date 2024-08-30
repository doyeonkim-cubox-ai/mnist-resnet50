import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # preprocess
    image = Image.open("./inference_data/image1.png")
    image = image.resize((28, 28)).convert('L')
    img_tensor = transforms.ToTensor()(image)
    img_tensor = img_tensor.unsqueeze(0)

    model = onnx.load("./model/model.onnx")
    session = ort.InferenceSession("./model/model.onnx", providers=['CPUExecutionProvider'])

    ort_in = {session.get_inputs()[0].name: np.array(img_tensor)}
    ort_out = session.run(None, ort_in)[0]

    # postprocess
    ort_out = ort_out.squeeze(0)
    res = np.argmax(ort_out)
    print(res)


if __name__ == "__main__":
    main()