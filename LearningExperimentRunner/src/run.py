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
import definitions as defs
import logging


##################################################
################## APP'S CONFIG ##################
LOG_LEVEL = logging.INFO
PORT = 5008
NSETS = 1


##################################################
############ CONSTANTS, DO NOT MODIFY ############
IP = '127.0.0.1'
CMD_EXP = ['gnome-terminal', '-x', 'roslaunch', 'pr2_grasping', 'all.launch']
CMD_UI = ['rviz:=true', 'gui:=true']
CMD_MON = 'exec python ./src/experiment_monitor_node.py '
ROS_APP_OUTPUT_DIR = 'output/'
EXP_RESULTS_DIR = 'results/'


##################################################
def getPackageDir():
	pkgDir =subprocess.check_output(['rospack', 'find', 'pr2_grasping'])
	return pkgDir.replace('\n', '/')


##################################################
def cleanOutputDir(pkgDir_):
	outputDir = pkgDir_ + ROS_APP_OUTPUT_DIR
	if os.path.exists(outputDir):
		shutil.rmtree(outputDir)
		os.mkdir(outputDir)

	if os.path.exists(defs.MONITOR_OUTPUT):
		shutil.rmtree(defs.MONITOR_OUTPUT)
		os.mkdir(defs.MONITOR_OUTPUT)


##################################################
def getExpDestination():
	# create results directory
	subprocess.call(['mkdir','-p', EXP_RESULTS_DIR])

	# check the destination directory
	experiment = 1
	while(True):
		destination = 'exp' + str(experiment)

		# check if the experiment number is already used
		change = False
		for f in os.listdir(EXP_RESULTS_DIR):
			if destination in f:
				experiment = experiment + 1
				change = True
				break

		# if no change was done, then this is the right directory name
		if not change:
			break

	return (EXP_RESULTS_DIR + destination + '_' + defs.STAMP_FORMAT + '/').format(datetime.datetime.now())


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

	files = os.listdir(defs.MONITOR_OUTPUT)
	for f in files:
		filename = os.path.join(defs.MONITOR_OUTPUT, f)
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
	formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
	logger = logging.getLogger('RUNNER')
	logger.setLevel(LOG_LEVEL)
	logFilename = EXP_RESULTS_DIR + 'runner.log'
	fh = logging.FileHandler(logFilename, 'w')
	ch = logging.StreamHandler()
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	logger.addHandler(fh)
	logger.addHandler(ch)


	# check the right version of python
	if int(sys.version[0]) != 2:
		logger.info('  ERROR: required Python v2 (>= 2.7.3)\n')

	else:
		try:
			# check if enough args were given
			if (len(sys.argv) < 4):
				print('\nNOT ENOUGH ARGUMENTS GIVEN.')
				print('   Usage: python run.py <catkin_workspace> <worlds_list_file> <show_ui>\n')
				sys.exit(0)

			catkinDir = checkDirName(sys.argv[1])
			worldsList = sys.argv[2]
			rviz = sys.argv[3].lower() == 'true'
			packageDir = getPackageDir()

			logger.info('Cleaning output directory...')
			cleanOutputDir(packageDir)
			resultsDest = getExpDestination()


			# set the port to listen the experiment monitor node
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.bind((IP, PORT))
			sock.listen(1)

			logger.info('*** Beginning learning experiments ***')

			# run a experiment with each world
			with open(worldsList) as worlds:
				for world in worlds:
					world = world.replace('\n', '')
					logger.debug('...read line "' + world + '"')

					# ignore commented lines
					if len(world) == 0 or world[0] == '#':
						continue

					world = world.split('#')[0].replace(' ', '')
					logger.info('========================================')
					logger.info('Evaluating world "' + world + '"')


					for k in range(1):

						# clear the log folder before every experiment
						loggingPath = os.path.expanduser('~/.ros/log/')
						if os.path.exists(loggingPath):
							shutil.rmtree(loggingPath)

						logger.info('...launching ROS')
						cmd = CMD_EXP + ['world:=' + world]
						if (rviz):
							cmd = cmd + CMD_UI
						expProcess = subprocess.Popen(cmd, cwd=catkinDir, stderr=subprocess.STDOUT)
						logger.info('********** pid: ' + str(expProcess.pid) + ' **********')

						# little sleep to allow roscore to come up
						time.sleep(10)

						# start monitoring the experiments
						logger.info('...launching monitor node')
						cmd = CMD_MON + IP + ' ' + str(PORT) + ' ' + str(NSETS) + ' ' + world
						monitorProcess = subprocess.Popen(cmd, cwd='.', shell=True, stderr=subprocess.STDOUT)

						# wait for the monitor node to speak
						logger.info('...waiting monitor reply')
						data = None
						connection, addr = sock.accept()
						while True:
							data = connection.recv(128)
							logger.info('...rcvd: ' + data + ' (' + defs.finishString(int(data)) + ')')
							break
						connection.close()

						# kill the experiment once the monitor has talked
						logger.info('...sending signal to process')
						expProcess.send_signal(signal.SIGTERM)
						logger.info('...signal sent')
						time.sleep(5)

						# monitor the running process
						while expProcess.poll() is None:
							time.sleep(2)

						# copy experiment results
						copyResults(packageDir + ROS_APP_OUTPUT_DIR, resultsDest)

						if data == str(defs.EXP_DONE):
							break;
						else:
							logger.info('\t...experiment failed (attempt %d)', k)


					logger.info('\t...world "' + world + '" done')
					logger.info('\t...waiting for system to be ready')
					time.sleep(30)
					logger.info('========================================')


			# create the results dir in case is not already created
			subprocess.call(['mkdir','-p', resultsDest])
			shutil.move(logFilename, resultsDest)


			logger.info('*** Learning experiments execution finished ***')

		except IOError as e:
			logger.info('ERROR: ' + str(e))
