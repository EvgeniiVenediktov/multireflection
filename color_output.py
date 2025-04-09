RED = '\033[31m'
GREEN = '\033[1;32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\003[37m'
RESET = '\033[0m'
GRAY = '\x1b[1;30m'

def cprint(s, color):
    print(color + s + RESET)