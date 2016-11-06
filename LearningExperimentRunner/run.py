'''
@author: rodrigo
2016
'''
import os
import sys
import shutil
import subprocess
import time
import datetime
import signal
import socket


IP = '127.0.0.1'
PORT = 5008
CMD_EXP = 'exec roslaunch pr2_grasping all.launch world:='
CMD_MON = 'exec python ./experiment_monitor_node.py '
OUTPUT_DIR = 'output/'
RESULTS_DIR = 'results/' 


##################################################
def getPackageDir():
	pkgDir =subprocess.check_output(['rospack', 'find', 'pr2_grasping'])
	return pkgDir.replace('\n', '/')


##################################################
def cleanOutputDir(pkgDir_):
	outputDir = pkgDir_ + OUTPUT_DIR
	shutil.rmtree(outputDir)
	os.mkdir(outputDir)


##################################################
def getExpDestination():
	# create results directory
	subprocess.call(['mkdir','-p', RESULTS_DIR])

	# check the destination directory
	experiment = 1
	while(True):
		destination = 'exp' + str(experiment)

		# check if the experiment number is already used
		change = False
		for f in os.listdir(RESULTS_DIR):
			if destination in f:
				experiment = experiment + 1
				change = True
				break

		# if no change was done, then this is the right directory name
		if not change:
			break

	return RESULTS_DIR + destination + '_{:%Y-%m-%d_%H%M%S}/'.format(datetime.datetime.now())


##################################################
def copyResults(src_, dest_):
	# create results subdirectory
	subprocess.call(['mkdir','-p', dest_])

	# copy files
	files = os.listdir(src_)
	for f in files:
		filename = os.path.join(src_, f)
		if (os.path.isfile(filename)):
			shutil.move(filename, dest_)


##################################################
def checkDirName(catkinDir_):
	if (catkinDir_[len(catkinDir_) - 1] != '/'):
		return catkinDir_ + '/'
	else:
		return catkinDir_


##################################################
if __name__ == '__main__':
	# check the right version of python
	if int(sys.version[0]) != 2:
		print('  ERROR: required Python v2 (>= 2.7.3)\n')

	else:
		try:
			# check if enough args were given
			if (len(sys.argv) < 3):
				print('NOT ENOUGH ARGUMENTS GIVEN.\n')
				print('   Usage: python run.py <catkin_workspace_location> <worlds_list_file>')
				sys.exit(0)

			catkinDir = checkDirName(sys.argv[1])
			worldsList = sys.argv[2]
			packageDir = getPackageDir()

			print('Cleaning output directory...')
			# cleanOutputDir(packageDir)
			resultsDest = getExpDestination()

			copyResults(packageDir + OUTPUT_DIR, resultsDest)
			sys.exit(0)


			# set the port to listen the experiment monitor node
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.bind((IP, PORT))
			sock.listen(1)

			print('*** Beginning learning experiments ***')

			# run a experiment with each world
			with open(worldsList) as worlds:
				for world in worlds:
					world = world.replace('\n', '')
					print('\t ==> evaluating world "' + world + '"')


					print('\t...launching ROS\n')
					print('========================================')
					experimentProcess = subprocess.Popen(CMD_EXP + world, cwd=catkinDir, shell=True, stderr=subprocess.STDOUT)
					print('********** pid: ' + str(experimentProcess.pid) + ' **********')
					print('========================================\n')


					# little sleep to allow roscore to come up
					time.sleep(10)

					# start monitoring the experiments
					print('\t...launching monitor node')
					monitorProcess = subprocess.Popen(CMD_MON + IP + ' ' + str(PORT), cwd='.', shell=True, stderr=subprocess.STDOUT)

					# wait for the monitor node to speak
					connection, addr = sock.accept()
					while True:
						data = connection.recv(128)
						break

					# kill the experiment once the monitor has talked
					print('...sending SIGINT to process')
					experimentProcess.send_signal(signal.SIGINT)
					print('...signal sent')
					time.sleep(5)


					# monitor the running process
					while experimentProcess.poll() is None:
						time.sleep(2)

					print('\t...world "' + world + '" done')
					print('\t...waiting for system to be ready\n')
					time.sleep(5)


			connection.close()
			print('*** Learning experiments execution finished ***')


		except IOError as e:
			print('ERROR: ' + str(e))
