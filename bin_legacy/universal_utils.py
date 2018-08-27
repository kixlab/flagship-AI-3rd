import time
import os
import logging
from tqdm import tqdm, trange
from pytz import timezone, utc
from datetime import datetime

def get_current_datetime():
	local = timezone("Asia/Seoul")
	ts = time.time()
	return datetime.fromtimestamp(ts, local).strftime('%Y-%m-%d %H:%M:%S')

def print_dt(s):
	print('[%s] ' % get_current_datetime() + s)

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

def write_file(data, filename, json=False, joiner=os.linesep):
	with open(filename, 'w') as writefile:
		if json:
			json.dump(data, writefile, ensure_ascii=False)
		else:
			writefile.write(joiner.join(data))

def get_base_path():
	bin_path = os.path.dirname( os.path.abspath( __file__ ) )
	base_path = os.path.abspath(os.path.join(bin_path, os.pardir))
	return base_path

# Console: INFO ~
# Log file: DEBUG ~
def get_custom_logger(name, log_path):
	# logger 인스턴스를 생성 및 로그 레벨 설정
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)

	# fileHandler와 StreamHandler를 생성
	final_log_path = log_path + '.log' if not log_path.endswith('.log') else log_path
	fileHandler = logging.FileHandler(final_log_path)
	streamHandler = logging.StreamHandler()

	fileHandler.setLevel(logging.DEBUG)
	streamHandler.setLevel(logging.INFO)

	log_format = '[%(asctime)s:%(lineno)03s] %(message)s'
	formatter = logging.Formatter(log_format)

	def customTime(*args):
		utc_dt = utc.localize(datetime.utcnow())
		my_tz = timezone("Asia/Seoul")
		converted = utc_dt.astimezone(my_tz)
		return converted.timetuple()

	logging.Formatter.converter = customTime

	fileHandler.setFormatter(formatter)
	streamHandler.setFormatter(formatter)

	# Handler를 logging에 추가
	logger.addHandler(fileHandler)
	logger.addHandler(streamHandler)

	return logger