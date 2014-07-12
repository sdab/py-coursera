#  I got the base for this script from Ioura Batugowski in the fall '11 ML class and made a few changes for them
# to work with the latest ML course, along with some additional input from Ioura

from getpass import getpass
import sha
import random
from urllib import urlencode
from urllib2 import urlopen
from random import sample
import os.path
from numpy import *
import pdb
import json
from base64 import b64encode
from contextlib import closing

__all__ = ['submit']

# ================== CONFIGURABLES FOR EACH HOMEWORK ==================

challenge_url = 'https://class.coursera.org/ml-006/assignment/challenge'
submit_url = 'https://class.coursera.org/ml-006/assignment/submit'
homework_id = '6'

part_names = [
    'Gaussian Kernel',
    'Parameters (C, sigma) for Dataset 3',
    'Email Preprocessing',
    'Email Feature Extraction'
    ]

srcs = [
    'gaussianKernel.py',
    'dataset3Params.py',
    'processEmail.py',
    'emailFeatures.py',
    ]

def output(part_id, auxstring):
    x1 = sin(arange(1,11))
    x2 = cos(arange(1,11))
    ec = 'the quick brown fox jumped over the lazy dog'
    wi = 1 + abs((x1 * 1863).round()).astype(int)
    wi = hstack((wi, wi))

    fname = srcs[part_id-1].rsplit('.',1)[0]
    mod = __import__(fname, fromlist=[fname], level=1)
    func = getattr(mod, fname)

    if part_id == 1:
        return sprintf('%0.5f ', func(x1, x2, 2.))
    elif part_id == 2:
        from scipy.io import loadmat
        ex6data3 = loadmat('ex6data3.mat')
        X = ex6data3['X']
        y = ex6data3['y'].ravel().astype(int8)
        Xval = ex6data3['Xval']
        yval = ex6data3['yval'].ravel().astype(int8)
        C, sigma = func(X, y, Xval, yval)
        return sprintf('%0.5f ', [C, sigma])
    elif part_id == 3:
        return sprintf('%0.5f ', func(ec))
    elif part_id == 4:
        return sprintf('%0.5f ', func(wi))


# ============================== SUBMIT ===============================

def submit(part_id=None):
    print '==\n== [ml-class] Submitting Solutions | Programming Exercise %s\n==', \
          homework_id

    if part_id is None:
        part_id = prompt_part()

    if not is_valid_part(part_id):
        print '!! Invalid homework part selected.'
        print '!! Expected an integer from 1 to %d.' % (len(part_names)+1)
        print '!! Submission Cancelled'
        return

    login, password = login_prompt()
    if not login:
        print '!! Submission Cancelled'
        return

    submit_parts = [part_id] if part_id <= len(part_names) else range(1,len(part_names)+1)

    print '\n== Connecting to ml-class ... '

    for part_id in submit_parts:
        # Submit this part
        # Get Challenge
        login, ch, signature, auxstring = get_challenge(login, part_id)
        if not login or not ch or not signature:
            # Some error occured, error string in first return element.
            print '\n!! Error: %s\n' % login
            return

        ch_resp = challenge_response(login, password, ch)
        result, s = submit_solution(login, ch_resp, part_id, output(part_id, auxstring), source(part_id), signature)
        print '\n== [ml-class] Submitted Homework %s - Part %d - %s' % (
                homework_id, part_id, part_names[part_id-1])
        print '== %s' % s.strip()

# ============================== HELPERS ==============================

def sprintf(fmt, arg):
    "emulates (part of) Octave sprintf function"
    if isinstance(arg, tuple):
        # for multiple return values, only use the first one
        arg = arg[0]

    if isinstance(arg, (ndarray, list)):
        # concatenates all elements, column by column
        return ''.join(fmt % e for e in asarray(arg).ravel('F'))
    else:
        return fmt % arg

def prompt_part():
    print '== Select which part(s) to submit:'
    for i, name in enumerate(part_names):
        print '==   %d) %s [%s]' % (i+1, name, srcs[i])
    print '==   %d) All of the above\n==' % (len(part_names)+1)
    selpart = raw_input('Enter your choice [1-%d]:' % (len(part_names)+1))
    try:
        return int(selpart)
    except ValueError:
        return -1

def is_valid_part(part_id):
    return part_id and 1 <= part_id <= len(part_names)+1

def login_prompt():
    login = raw_input('login (Email address): ')
    password = getpass('Password: ')
    return login, password

def challenge_response(email, passwd, challenge):
    return sha.new(challenge + passwd).hexdigest()

def source(part_id):
    fname = srcs[part_id-1]
    try:
        # try relative path
        f = open(fname)
    except IOError:
        # else try the directory of this script
        fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        f = open(fpath)
    try:
        src = f.read() + '||||||||'
    finally:
        f.close()
    return src

def get_challenge(email, part):
    params = {
        'email_address': email,
        'response_encoding' : 'json',
        'assignment_part_sid': homework_id +'-'+ str(part) }

    #pdb.set_trace()
    with closing(urlopen(challenge_url, urlencode(params))) as f:
        incoming = json.loads(f.read().strip())
        login = incoming['email_address']
        ch = incoming['challenge_key']
        signature = incoming['state']
        auxstring = incoming['challenge_aux_data']
        return login, ch, signature, auxstring

def submit_solution(email, ch_resp, part, output, source, signature):
    #pdb.set_trace()
    params = {
    'assignment_part_sid': homework_id +'-'+ str(part),
    'email_address': email,
    'submission': b64encode(output),
    'submission_aux': b64encode(source),
    'challenge_response': ch_resp,
    'state': signature }

    f = urlopen(submit_url, urlencode(params))
    try:
        return 0, f.read()
    finally:
        f.close()

if __name__ == '__main__':
    submit()
