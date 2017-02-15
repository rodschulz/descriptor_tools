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


APP_LOCATION = 'build/DenseEvaluator'
CONFIG_LOCATION = 'config/config_dense_evaluator.yaml'
RESULTS_LOCATION = './results/' 


##################################################
def getClusterNumber(fileName):
	try:
		with open(fileName, 'r') as configFile:
			config = yaml.load(configFile)
			return config['clustering']['clusterNumber']

	except IOError as e:
		print('Error reading ' + fileName + ': ' + str(e) + ')')
		raise(IOError('Cant get configured clusters number'))


##################################################
def getDestination(nclusters):
	# create results directory if it doesn't exists
	#os.makedirs(RESULTS_LOCATION, exist_ok=True)
	subprocess.call(['mkdir','-p', RESULTS_LOCATION])

	# check the destination directory
	experiment = 1
	while(True):
		destination = 'clusters' + str(nclusters) + '_exp' + str(experiment)

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
def moveOutputData(appDirectory, destination, inputLine):
	parts = inputLine.split('/')
	inputDir = parts[len(parts) - 2]
	inputFile = parts[len(parts) - 1].split('.')[0]

	source = appDirectory + 'output/'
	dest = destination + inputDir + '/' + inputFile + '/'

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

		# get the application's directory
		appDirectory = os.path.abspath(sys.argv[1]) + '/'
		# get number of clusters to be used in the reduction process
		nclusters = getClusterNumber(appDirectory + CONFIG_LOCATION)
		# get destination directory
		destination = getDestination(nclusters)

		# read the input and process the data
		FNULL = open(os.devnull, 'w')
		with open(sys.argv[2]) as clouds:
			for line in clouds:
				line = line.replace('\n', '')

				# skip empty lines
				if line == '':
					continue;

				print('Processing file: ' + line)
				
				# run the app and wait
				cmd = appDirectory + APP_LOCATION + ' ' + line + ' | tee DenseEvaluator.log; mv DenseEvaluator.log output/' 
				process = subprocess.Popen(cmd, cwd=appDirectory, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
				process.wait()

				if process.returncode == 0:
					print('\t...execution successful, copying results')
					moveOutputData(appDirectory, destination, line)
				else:
					print('\t...execution failed')

	except IOError as e:
		print('ERROR: ' + str(e))


##################################################
if __name__ == '__main__':
	# Check the right version
	if int(sys.version[0]) < 3:
		print('  ERROR: wrong Python version, required 3.x.x\n')
	else:
		main()

