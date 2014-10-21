import sys
from jester.util import add

def main(argv):
    a = int(argv[0])
    b = int(argv[1])
    print add(a,b)

if __name__ == '__main__':
    main(sys.argv[1:])
