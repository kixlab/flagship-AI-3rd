import os
import json
from tqdm import tqdm

# Read text file and return array of sentences
def load_text_file(filename, ignore_first=False, max_num=10000000, as_words=False):
	with open(filename, 'rb') as readfile:
		if ignore_first:
			lines = []
			for l in tqdm(readfile.readlines()[:max_num]):
				words = l.decode('utf8', 'ignore').strip().split(' ')
				if len(words) > 0 and words[0].startswith('[[') and words[0].endswith(']]'):
					lines.append(
						' '.join(words[1:]) if not as_words else words[1:])
				else:
					lines.append('')
		else:
			if not as_words:
				lines = [l.decode('utf8', 'ignore').strip()
                                    for l in readfile.readlines()]
			else:
				lines = [l.decode('utf8', 'ignore').strip().split(' ')
                                    for l in readfile.readlines()]
	return lines

def read_json(input_fn):
	with open(input_fn, 'r') as readfile:
		data = json.load(readfile)
	return data

def write_file(data, filename, is_json=False, joiner=os.linesep, csv=None):
	with open(filename, 'w') as writefile:
		if is_json:
			json.dump(data, writefile, ensure_ascii=False)
		else:
			writefile.write(joiner.join(data))
