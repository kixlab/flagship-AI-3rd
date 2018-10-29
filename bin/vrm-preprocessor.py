from utils.file import load_text_file, write_file
from collections import Counter

def convert_vrm_to_da():
  data = load_text_file('../dataset/vrm/vrm-single-vrm.txt')
  new_data = []
  for d in data:
    if d.endswith('D'):
      new_data.append('d')
    elif d.endswith('E'):
      new_data.append('e')
    elif d.endswith('K'):
      new_data.append('k')
    else:
      new_data.append('x')

  print(sorted(Counter(new_data).items()))
  # write_file(new_data, '../dataset/vrm/vrm-single-da.txt')

convert_vrm_to_da()
