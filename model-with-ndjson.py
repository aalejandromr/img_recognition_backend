from quickdraw import QuickDrawData, QuickDrawDataGroup
import random
import string
import os
from os import path

GROUPS = ['mailbox', 'birthday cake', 'waterslide']

def save_qd_image(qd_image, file_name, dir):
  if path.exists(dir) is not True:
    os.mkdir(dir)
  qd_image.save(f'{dir}/{file_name}')
  

def random_file_name():
  N = 7
  return ''.join(random.choices(string.ascii_lowercase + string.digits, k = N))

def load_drawings_locally():
  qd = QuickDrawData()
  qd.load_drawings(GROUPS)
  for group in GROUPS:
    qd_drawing_group = QuickDrawDataGroup(group).drawings
    for drawing in qd_drawing_group:
      save_qd_image(drawing.image, f'{random_file_name()}.png', f'training_photos/{group}')

if __name__ == '__main__':
  load_drawings_locally()