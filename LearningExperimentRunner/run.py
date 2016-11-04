'''
@author: rodrigo
2016
'''
import os
import sys

##################################################
def main():
	try:
		if (len(sys.argv) < 1):
			print('NOT ENOUGH ARGUMENTS GIVEN.\n')
			print('   Usage: python run.py\n\n')
			return


	except IOError as e:
		print('ERROR: ' + str(e))


##################################################
if __name__ == '__main__':
	# Check the right version
	if int(sys.version[0]) < 3:
		print('  ERROR: wrong Python version, required 3.x.x\n')
	else:
		main()
