import os
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', help='load_path', type=str)
args = parser.parse_args()


while 1:
    if os.path.isfile(args.load_path) is True:
        print("Ready to eval! -> {}".format(args.load_path))
        time.sleep(3)
        break
