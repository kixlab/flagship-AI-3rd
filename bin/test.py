# -*- coding: utf-8 -*-

import os
import csv
import tqdm
import logging
from pytz import timezone, utc
from datetime import datetime


# logger 인스턴스를 생성 및 로그 레벨 설정
logger = logging.getLogger('name')
logger.setLevel(logging.DEBUG)

# fileHandler와 StreamHandler를 생성
fileHandler = logging.FileHandler('./test.log')
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

# logging 
logger.debug("debug")
logger.info("info")
logger.warning("warning")
logger.error("error")
logger.critical("critical")