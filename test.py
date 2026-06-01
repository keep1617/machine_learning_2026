import sys
import select
import tty
import termios

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setcbreak(fd)

try:
    while True:
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 'a':
                print("a 눌렀다!")
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)