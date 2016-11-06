'''
@author: rodrigo
2016
'''
import os
import sys
import subprocess
import time
import thread
import signal
import socket


IP = '127.0.0.1'
PORT = 5008
CMD_EXP = 'exec roslaunch pr2_grasping all.launch world:='
CMD_MON = 'exec python ./LearningExperimentRunner/experiment_monitor_node.py '


##################################################
if __name__ == '__main__':
	# check the right version of python
	if int(sys.version[0]) != 2:
		print('  ERROR: required Python version 2.x.x\n')

	else:
		try:
			# check if enough args were given
			if (len(sys.argv) < 3):
				print('NOT ENOUGH ARGUMENTS GIVEN.\n')
				print('   Usage: python run.py <catkin_workspace_location> <worlds_list_file>')
				sys.exit(0)

			catkinDir = sys.argv[1]
			worldsFile = sys.argv[2]

			# set the port to listen the experiment monitor node
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.bind((IP, PORT))
			sock.listen(1)

			print('*** Beginning learning experiments ***')

			# run a experiment with each world
			with open(worldsFile) as worlds:
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
