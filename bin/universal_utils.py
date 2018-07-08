import time
import datetime
import pytz

def get_current_datetime():
	local = pytz.timezone("Asia/Seoul")
	ts = time.time()
	return datetime.datetime.fromtimestamp(ts, local).strftime('%Y-%m-%d %H:%M:%S')

def print_dt(s):
	print('[%s] ' % get_current_datetime() + s)

# Read text file and return array of sentences
def load_text_file(filename):
	with open(filename, 'rb') as readfile:
		lines = [l.decode('utf8', 'ignore').strip()
								 for l in readfile.readlines()]
	return lines