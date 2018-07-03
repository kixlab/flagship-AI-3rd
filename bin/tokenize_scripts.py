from os import listdir
from os.path import isfile, join, dirname
import re
import json
import zipfile

## Varaibles
read_zip_file = '../scripts/test.zip'
write_zip_file = '../results/test.zip'
read_directory = '../scripts'
write_directory = '../results'

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

dn = dirname(__file__)

with zipfile.ZipFile(read_zip_file, 'r') as z:
  for fn in z.namelist():
    print("Parsing file: %s" % fn)
    # read_path = join(dn, read_directory, fn)
    write_path = join(write_directory, fn)
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
        z.write(join(write_directory, f), arcname=f)
