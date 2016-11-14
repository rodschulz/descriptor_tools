'''
@author: rodrigo
2016
'''
import os
import sys
import socket
import re
import wnck
import time
import rospy
import definitions as defs
from std_msgs.msg import Int32
from sensor_msgs.msg import PointCloud2
from pr2_grasping.msg import EvaluationStatus


IP = '127.0.0.1'
PORT = 9999
NSETS = 1

experimentRunning = False


##################################################
def setsCallback(msg_):
	if msg_.data >= NSETS:
		rospy.loginfo('Experiment done (%d)', msg_.data)

		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((IP, PORT))
		sock.send(defs.EXP_DONE)
		sock.close()

		rospy.signal_shutdown('Experiment finished')


##################################################
def evalCallback(msg_):
	pattern1 = re.compile('.*Gazebo.*')
	pattern2 = re.compile('.*rviz.*')

	screen = wnck.screen_get_default()
	screen.force_update()
	windows = screen.get_windows()

	for w in windows:
		if pattern1.match(w.get_name()):
			os.system('import -window "' + w.get_name() + '" ~/Downloads/gazebo.png')
		elif pattern2.match(w.get_name()):
			os.system('import -window "' + w.get_name() + '" ~/Downloads/rviz.png')


##################################################
def cloudCallback(msg_):
	global experimentRunning
	experimentRunning = True


##################################################
def watchdogCallback(event_):
	if not experimentRunning:
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((IP, PORT))
		sock.send(str(defs.EXP_START_FAILED))
		sock.close()

		rospy.signal_shutdown('Experiment start failed')


##################################################
if __name__ == '__main__':
	try:
		if (len(sys.argv) < 3):
			print('NOT ENOUGH ARGUMENTS GIVEN.\n')
			print('   Usage: python experiment_monitor_node.py <ip> <port> <nsets>')
			sys.exit(0)

		IP = sys.argv[1]
		PORT = int(sys.argv[2])
		NSETS = int(sys.argv[3])

		rospy.init_node('experiment_monitor', anonymous=False)

		# set a watchdog for faulty starts
		rospy.Timer(rospy.Duration(120), watchdogCallback, oneshot=True)

		startTime = rospy.get_rostime()
		rospy.Subscriber('/pr2_grasping/processed_sets', Int32, setsCallback)
		rospy.Subscriber('/pr2_grasping/evaluation_status', EvaluationStatus, evalCallback)
		rospy.Subscriber('/pr2_grasping/labeled_cloud', PointCloud2, cloudCallback)

		rospy.spin()

	except rospy.ROSInterruptException as e:
		print('Node interrupted: ' + str(e))
		pass