import time
import datetime

def get_current_datetime():
	ts = time.time()
	return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def print_dt(s):
	print('[%s] ' % get_current_datetime() + s)

# Read text file and return array of sentences
def load_sentences(filename):
	with open(filename, 'rb') as readfile:
		sentences = [l.decode('utf8', 'ignore')
								 for l in readfile.readlines()]
	return sentences