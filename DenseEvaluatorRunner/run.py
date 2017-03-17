'''
@author: rodrigo
2016
'''
import os
import sys
import subprocess
import shutil
import yaml
import datetime
import logging


##################################################
################## APP'S CONFIG ##################
APP_LOCATION = 'build/DenseEvaluator'
CONFIG_LOCATION = 'config/config_dense_evaluator.yaml'
RESULTS_LOCATION = './results/' 
LOG_LEVEL = logging.INFO

##################################################
logger = None


##################################################
def computeSizeDCH(config_):
	nband = int(config_['bandNumber'])
	stat = config_['stat']

	if (stat == 'mean') or (stat == 'median'):
		return str(nband * int(config_['binNumber']))
	elif (stat == 'hist10') or (stat == 'hb10'):
		return str(nband * 18)
	elif (stat == 'hist20') or (stat == 'hb20'):
		return str(nband * 9)
	else:
		return 'xx'


##################################################
def getDescriptorData(fileName_):
	try:
		with open(fileName_, 'r') as configFile:
			config = yaml.load(configFile)

			ncluster = config['clustering']['clusterNumber']
			dtype = config['descriptor']['type']
			dsize = 0

			if dtype == 'DCH':
				dsize = computeSizeDCH(config['descriptor']['DCH'])
			elif dtype == 'SHOT':
				dsize = '352'
			elif dtype == 'PFH':
				dsize = '125'
			elif dtype == 'FPFH':
				dsize = '33'
			elif dtype == 'SpinImage':
				dsize = '153'


			return [ncluster, dtype, dsize]

	except IOError as e:
		logger.error('Error reading ' + fileName + ': ' + str(e) + ')')
		raise(IOError('Cant get configured clusters number'))


##################################################
def getDestination(nclusters_, dtype_, dsize_):
	# create results directory if it doesn't exists
	subprocess.call(['mkdir','-p', RESULTS_LOCATION])

	# check the destination directory
	experiment = 1
	while(True):
		destination = 'clusters' + str(nclusters_) + '_' + dtype_ + dsize_ + '_exp' + str(experiment)

		# check if the experiment number is already used
		change = False
		for f in os.listdir(RESULTS_LOCATION):
			if destination in f:
				experiment = experiment + 1
				change = True
				break

		# if no change was done, then this is the right directory name
		if not change:
			break

	return RESULTS_LOCATION + destination + '_{:%Y-%m-%d_%H%M%S}/'.format(datetime.datetime.now())


##################################################
def moveOutputData(appDir_, destination_, inputLine_):
	parts = inputLine_.split('/')
	inputDir = parts[len(parts) - 2]
	inputFile = parts[len(parts) - 1].split('.')[0]

	source = appDir_ + 'output/'
	dest = destination_ + inputDir + '/' + inputFile + '/'

	shutil.copytree(source, dest)
	shutil.rmtree(source)
	os.mkdir(source)


##################################################
def main():
	try:
		if (len(sys.argv) < 3):
			print('NOT ENOUGH ARGUMENTS GIVEN.\n')
			print('   Usage: python run.py <app_dir_location> <cloud_list_location>\n\n')
			return

		# setup logging
		formatter = logging.Formatter('[%(levelname)-5s] %(message)s')
		logger = logging.getLogger('EXTRACTOR')
		logger.setLevel(LOG_LEVEL)
		ch = logging.StreamHandler()
		ch.setFormatter(formatter)
		logger.addHandler(ch)


		# get the application's directory
		appDirectory = os.path.abspath(sys.argv[1]) + '/'
		# get data about the descriptor
		nclusters, dtype, dsize = getDescriptorData(appDirectory + CONFIG_LOCATION)
		# get destination directory
		destination = getDestination(nclusters, dtype, dsize)

		# read the input and process the data
		FNULL = open(os.devnull, 'w')
		with open(sys.argv[2]) as clouds:
			for line in clouds:
				line = line.replace('\n', '')

				# skip empty lines
				if line == '':
					continue;

				logger.info('Processing file: ' + line)
				
				# run the app and wait
				cmd = appDirectory + APP_LOCATION + ' ' + line + ' | tee DenseEvaluator.log; mv DenseEvaluator.log output/' 
				process = subprocess.Popen(cmd, cwd=appDirectory, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
				process.wait()

				if process.returncode == 0:
					logger.info('\t...execution successful, copying results')
					moveOutputData(appDirectory, destination, line)
				else:
					logger.warn('\t...execution failed')

	except IOError as e:
		logger.error('ERROR: ' + str(e))


##################################################
if __name__ == '__main__':
	# Check the right version
	if int(sys.version[0]) < 3:
		print('  ERROR: wrong Python version, required 3.x.x\n')
	else:
		main()

