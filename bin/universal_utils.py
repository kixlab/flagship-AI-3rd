import time
import os
import logging
from pytz import timezone, utc
from datetime import datetime

def get_current_datetime():
	local = timezone("Asia/Seoul")
	ts = time.time()
	return datetime.fromtimestamp(ts, local).strftime('%Y-%m-%d %H:%M:%S')

def print_dt(s):
	print('[%s] ' % get_current_datetime() + s)

# Read text file and return array of sentences
def load_text_file(filename):
	with open(filename, 'rb') as readfile:
		lines = [l.decode('utf8', 'ignore').strip()
								 for l in readfile.readlines()]
	return lines

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
	fileHandler = logging.FileHandler(log_path)
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