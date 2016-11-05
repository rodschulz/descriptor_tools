'''
@author: rodrigo
2016
'''
import os
import sys
import subprocess
import time
# import rospy as rp


cmd = 'roslaunch pr2_grasping all.launch world:='
catkinDir = None


##################################################
def monitorExperiments():
	return


##################################################
if __name__ == '__main__':
	# check the right version of python
	if int(sys.version[0]) < 3:
		print('  ERROR: wrong Python version, required 3.x.x\n')

	else:
		try:
			# check if enough args were given
			if (len(sys.argv) < 3):
				print('NOT ENOUGH ARGUMENTS GIVEN.\n')
				print('   Usage: python run.py <catkin_workspace_location> <worlds_list_file>\n\n')
				sys.exit(0)


			catkinDir = sys.argv[1]
			worldsFile = sys.argv[2]

			print('*** Beginning learning experiments ***')


			# run a experiment with each world
			with open(worldsFile) as worlds:
				for world in worlds:
					world = world.replace('\n', '')
					print('...evaluating world "' + world + '"\n\n')


					print('...launching ROS\n====================\n')
					process = subprocess.Popen(cmd + world, cwd=catkinDir, shell=True, stderr=subprocess.STDOUT)


					# monitor the running process
					secs = 30
					while process.poll() is None:
						print('.....process running, sleeping ' + str(secs) + ' secs')
						time.sleep(secs)


				print('...world "' + world + '"done')

			print('*** Learning experiments execution finished ***')

		except IOError as e:
			print('ERROR: ' + str(e))
