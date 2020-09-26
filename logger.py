# use this log module to globally disable when needed
import sys
import json
import traceback

def debug(s):
    meta = traceback.format_stack()[0].split('\n')[0].strip().replace('File', '', 1).replace(', line ', ':', 1)
    if type(s) == type({}):
        s = json.dumps(s, indent=2)
    sys.stdout.write('\033[34m[DEBUG,%s]\033[39m %s\n' % (meta, s))

def error(s):

    meta = traceback.format_stack()[0].split('\n')[0].strip().replace('File', '', 1).replace(', line ', ':', 1)
    if type(s) == type({}):
        s = json.dumps(s, indent=2)
    sys.stderr.write('\033[31m[ERROR,%s]\033[39m %s\n' % (meta, s))
    