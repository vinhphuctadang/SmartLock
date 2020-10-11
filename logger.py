# use this log module to globally disable when needed
import sys
import json
import traceback

def debug(*args, **kargs):
    meta = traceback.format_stack()[0].split('\n')[0].strip().replace('File', '', 1).replace(', line ', ':', 1)
    print('\033[34m[DEBUG,%s]\033[39m' % meta, *args, file=sys.stdout, **kargs)

def error(*args, **kargs):
    meta = traceback.format_stack()[0].split('\n')[0].strip().replace('File', '', 1).replace(', line ', ':', 1)
    print('\033[31m[ERROR,%s]\033[39m' % meta, *args, file=sys.stderr, **kargs)