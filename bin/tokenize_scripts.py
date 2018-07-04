from os import listdir, remove, makedirs
from os.path import isfile, join, dirname, exists, basename
import re
import json
import zipfile

## Varaibles
# All of directories must exist
read_zip_file = '../scripts/test2.zip'
write_zip_file = '../results/test2.zip'
# read_directory = '../scripts'
write_directory = '.'  # For temp files to be zipped

log_per = 3

## Regex for tokenizing words
token_re = re.compile('([0-9]+[.][0-9]+|[0-9A-Za-zㄱ-ㅎㅏ-ㅢ가-힣]+|\.\.+|\.|\?|\!|\,)')
in_brackets_re = re.compile('\(.*?\)')  # Limitation on nested brackets like '(a(b)c)'

## Constants


## Functions
def tokenize_sentence(s):
  # Remove words in brackets
  result = in_brackets_re.sub(' ', s)

  # Divide sentence into words with regex
  result = token_re.findall(result)
  return result


## Main

with zipfile.ZipFile(read_zip_file, 'r') as z:
  print("Open script ZIP file")
  zip_info = z.infolist()

  for idx, zi in enumerate(zip_info):
    fn = zi.filename
    if (not fn.endswith('.json')):
      continue

    if (idx + 1) % log_per == 0:
      print("Parsing file %6d: %s" % (idx+1, fn))
    write_path = join(write_directory, basename(fn))
    result_json = []

    with z.open(fn, 'r') as readfile:
      with open(write_path, 'w', encoding='utf8') as writefile:
        dialogs = json.load(readfile)
        result = []
        for d in dialogs:
          result_dialog = {'context': d['context']}
          result_dialog['lines'] = []
          for l in d['lines']:
            result_line = {'speaker': l['speaker'], 'message': l['message']}
            result_line['tokens'] = tokenize_sentence(l['message'])
            result_dialog['lines'].append(result_line)
          result.append(result_dialog)
        json.dump(result, writefile, ensure_ascii=False)

with zipfile.ZipFile(write_zip_file, 'w') as z:
  file_names = [f for f in listdir(write_directory) if (isfile(join(write_directory, f)) and f.endswith('.json'))]
  for f in file_names:
    f_path = join(write_directory, f)
    z.write(f_path, arcname=f)
    remove(f_path)
