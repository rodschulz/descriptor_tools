'''
@author: rodrigo
2016
'''
import os
import sys
# import rospy as rp


launchCmd = 'roslaunch pr2_grasping all.launch'
catkinDir = None


##################################################
if __name__ == '__main__':
	# Check the right version
	if int(sys.version[0]) < 3:
		print('  ERROR: wrong Python version, required 3.x.x\n')
	else:
		try:
			# Check if enough args were given
			if (len(sys.argv) < 3):
				print('NOT ENOUGH ARGUMENTS GIVEN.\n')
				print('   Usage: python run.py <catkin_workspace_location> <worlds_list_file>\n\n')
				sys.exit(0)

			catkinDir = sys.argv[1]
			worldsFile = sys.argv[2]

			# Run a experiment with each of the listed worlds
			with open(worldsFile) as worlds:
				for line in worlds:
					line = line.replace('\n', '')
					print(line)

		except IOError as e:
			print('ERROR: ' + str(e))
