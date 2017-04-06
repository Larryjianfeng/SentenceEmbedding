#encoding=utf-8
import sys, os
sys.path.append('/data1/yanjianfeng/fastText+LSH')
from segs import segs
import pickle

if __name__ == '__main__':
	if len(sys.argv)  != 3:
		print 'type in the name of input and output'
	else:
		d = open(sys.argv[1], 'r').readlines()
		d = [segs(i.decode('utf-8')).encode('utf-8') for i in d]
		d = [i for i in d if i != '']
		with open(sys.argv[2], 'w') as f:
			for i in d:
				if i.split(' ') >= 2:
					f.write(i + '\n')
		print 'finished'