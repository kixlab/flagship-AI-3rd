# -*- coding: utf-8 -*-

from konlpy.tag import Kkma

kkma = Kkma()

x = "그리고 그까짓 시험문제 몇 개 틀린 건 문제가 아녜요 집안에 어른들 계셔서 혹시 아이들이 울면 걱정하실 까봐 그냥 웬만한 건 덮어주며 지나가니깐 이젠 공부하기 싫어두 아버님 방으루 건너가구- 혼날 일 저질러 놓면 어머님 방에 숨어서 안 나오려 하지 뭐에요"
for s in kkma.sentences(x):
  print(s)
