#!/bin/python

import sys


#print(','.join([a + ":"+ str(index) for index,a in enumerate(sys.argv[1].split(","))]))
print(','.join([a + ":0" for index,a in enumerate(sys.argv[1].split(","))]))

