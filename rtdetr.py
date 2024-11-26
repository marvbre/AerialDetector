"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torchvision.transforms as T
import numpy as np 
import onnxruntime as ort 
from PIL import Image, ImageDraw

class RTDETR:

   def __init__(self, path_to_onnx):
       self.session = ort.InferenceSession(path_to_onnx)
       print("Inference is performed on ", ort.get_device())

   def predict(self, image):

      if isinstance(image, str):
         image = Image.open(image).convert('RGB')
         w, h = image.size
         orig_size = torch.tensor([w, h])[None]
      elif isinstance(image, np.array):
         image = image
         w, h, _ = image.shape
         orig_size =  torch.tensor([w, h])[None]
      elif isinstance(image, torch.Tensor):
         image = image
         orig_size = image.shape
      elif isinstance(image, torch.cuda.Tensor):
         image = image
         orig_size = image.shape
      else:
          print("Wrong image format.")

      transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
      t_image = transforms(image)[None]

      output = self.session.run(output_names=None, input_feed={'images': t_image.data.numpy(), "orig_target_sizes": orig_size.data.numpy()})
      return output # labels, boxes, scores = output
      #output = self.session.run(output_names=['labels', 'boxes', 'scores'], input_feed={'images': image, "orig_target_sizes": orig_size.data.numpy()})

   def draw(images, labels, boxes, scores, thrh = 0.6):
      """Called like this : draw([im_pil], labels, boxes, scores) """

      for i, im in enumerate(images):
         draw = ImageDraw.Draw(im)

         scr = scores[i]
         lab = labels[i][scr > thrh]
         box = boxes[i][scr > thrh]

         for b in box:
               draw.rectangle(list(b), outline='red',)
               draw.text((b[0], b[1]), text=str(lab[i].item()), fill='blue', )

         im.save(f'results_{i}.jpg')
