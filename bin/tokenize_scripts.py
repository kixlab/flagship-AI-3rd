from os import listdir, remove, makedirs, rename
from os.path import isfile, join, dirname, exists, basename
import re
import json
import zipfile
import shutil

## Varaibles
# All of directories must be in advance
read_zip_file = '../scripts/scripts.zip'
write_zip_file = '../results/result.zip'
write_directory = '../results/scripts'
tmp_directory = '../results/tmp'  # For temp files to be zipped

start_file_num = 1  # index starts with 1
save_num = 500

## Regex for tokenizing words
token_re = re.compile('([0-9]+[.][0-9]+|[0-9A-Za-zㄱ-ㅎㅏ-ㅢ가-힣]+|\.\.+|\.|\?|\!|\,)')
in_brackets_re = re.compile('\(.*?\)')  # Limitation on nested brackets like '(a(b)c)'



## Functions
def tokenize_sentence(s):
  # Remove words in brackets
  result = in_brackets_re.sub(' ', s)

  # Divide sentence into words with regex
  result = token_re.findall(result)
  return result


## Main
print('Start tokenizing!')
with zipfile.ZipFile(read_zip_file, 'r') as z:
  print("Open script ZIP file")

  for idx, fn in enumerate(sorted(z.namelist()), 1):
    if (idx >= start_file_num and fn.endswith('.json')):
      write_path = join(tmp_directory, basename(fn))
      result_json = []

      with z.open(fn, 'r') as readfile:
        try:
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
          if len(result) > 0:
            with open(write_path, 'w', encoding='utf8') as writefile:
              json.dump(result, writefile, ensure_ascii=False)
        except:
          print('JSON is Empty!')

    if (idx % save_num == 0):
      tmp_files = listdir(tmp_directory)
      for f in tmp_files:
        shutil.move(join(tmp_directory, f), join(write_directory, f))
      print("Files are moved from tmp to dest folder; step %5d" % idx)

  for f in listdir(tmp_directory):
    tmp_files = listdir(tmp_directory)
    for f in tmp_files:
      shutil.move(join(tmp_directory, f), join(write_directory, f))
  print("Move the last files from tmp to dest folder!")

# with zipfile.ZipFile(write_zip_file, 'w') as z:
#   file_names = [f for f in listdir(write_directory) if (isfile(join(write_directory, f)) and f.endswith('.json'))]
#   for f in file_names:
#     f_path = join(write_directory, f)
#     z.write(f_path, arcname=f)
#     remove(f_path)
