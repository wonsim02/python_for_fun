# -*- coding: utf-8 -*-
from tkinter import Tk, Frame, StringVar, Label, FALSE
import tkinter.font as tkFont

DELIM = "," # 상태의 이름을 구분함
DELIM_CODE = "\n\n"
DELIM_ESC = "\r\n"
EMPTY = ""

MAX_LENGTH = 9

SETTINGS1 = "settings1.txt"
SETTINGS2 = "settings2.txt"
SETTINGS3 = "settings3.txt"
SETTINGS4 = "settings4.txt"

# 상태 클래스
class State:
    # 초기화
    def __init__(self, name, delta, out):
        self.__name = name # 상태 이름
        self.__dict = delta.copy() # 상태 변환 함수
        self.__lambda = out.copy() # 출력 함수

    # 상태 변환
    def transition(self, char):
        return self.__dict[char]

    def get_output(self, char):
        return self.__lambda[char]

    def __str__(self):
        return self.__name

    __repr__ = __str__

# Mealy Machine 함수를 구현해 반환한다
def get_mealy(states, inputs, outputs, func, out, start):
    state_dict = {} # 상태 목록
    dict_temp = {}
    temp_dict = {}
    # 상태 객체 생성 및 설정
    for i in range(0, len(states)):
        for j in range(0, len(inputs)):
            dict_temp[inputs[j]] = func[i][j] # 상태변환 함수 설정
            temp_dict[inputs[j]] = int(out[i][j]) # 출력함수 설정
        state_dict[states[i]] = State(states[i], dict_temp, temp_dict) # 상태 객체 생성 후 저장

    # Mealy Machine 함수 구현
    def mealy(chars):
        current = state_dict[start] # 현재 상태 = 초기 상태
        while 1:
            while not len(chars) is 0:
                char = chars.pop(0)
                if char is EMPTY:
                    continue
                elif char in DELIM_ESC:
                    return
                elif char in inputs:
                    exec(outputs[current.get_output(char)])
                    next_state = current.transition(char) # 상태 변화 함수의 결과값
                    current = state_dict[next_state] # 상태를 바꾼다
            root.update()

    return mealy # 함수를 반환한다

chars = []

def keyinput(e):
    chars.append(e.char)

def display(s):
    if len(s) <= MAX_LENGTH:
        v.set(s)
    else:
        v.set(s[-MAX_LENGTH:])

# Mealy Machine 입력
f = open(SETTINGS1, "r")
temp = f.read().split("\n")
states = temp[0].split(DELIM)
start = temp[1]
inputs = temp[2]
f.close()
f = open(SETTINGS2, "r")
outputs = f.read().split(DELIM_CODE)
f.close()
f = open(SETTINGS3, "r")
temp = f.read().split("\n")
func = []
index = 0
for i in range(0, len(states)):
    func.append(temp[index].split(DELIM))
    index = index + 1
out = []
for i in range(0, len(states)):
    out.append(temp[index].split(DELIM))
    index = index + 1
f.close()
f = open(SETTINGS4, "r", encoding = "UTF-8")
initial = f.read()
f.close()

root = Tk()
frame = Frame(root, width = 486, height = 60)
root.resizable(width=FALSE, height=FALSE)
frame.bind("<Key>", keyinput)
frame.pack_propagate(0)
frame.pack()
v = StringVar()
font = tkFont.Font(root = root, name = "Gulim", size = 40)
Label(frame, textvariable = v, font = font).pack(side = "left")
frame.focus_set()

mealy = get_mealy(states, inputs, outputs, func, out, start)
exec(initial)
print("========== Successfuly Loaded ==========")
while 1:
    mealy(chars)
