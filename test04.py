import winsound

# 定义音符和持续时间
notes = {
    "C": 262,  # do
    "D": 294,  # re
    "E": 330,  # mi
    "F": 349,  # fa
    "G": 392,  # sol
    "A": 440,  # la
    "B": 494   # si
}

# 定义音乐
music = [
    ("E", 500),  # mi
    ("E", 500),  # mi
    ("F", 500),  # fa
    ("G", 500),  # sol
    ("G", 500),  # sol
    ("F", 500),  # fa
    ("E", 500),  # mi
    ("D", 500),  # re
    ("C", 500),  # do
    ("C", 500),  # do
    ("D", 500),  # re
    ("E", 500),  # mi
    ("E", 500),  # mi
    ("D", 500),  # re
    ("D", 500)   # re
]

# 播放音乐
for note, duration in music:
    frequency = notes[note]
    winsound.Beep(frequency, duration)
