'''
@author: rodrigo
2016
'''
import os
import sys
import socket
import re
import wnck
import datetime
import rospy
import definitions as defs
from std_msgs.msg import Int32
from sensor_msgs.msg import PointCloud2
from pr2_grasping.msg import EvaluationStatus
from moveit_msgs.msg import PickupActionGoal


##################################################
################# NODE'S CONFIG ##################
SCREENSHOT_BEFORE = True
SCREENSHOT_EVAL = True
SCREENSHOT_AFTER = False

SCREENSHOT_GAZEBO = True
SCREENSHOT_RVIZ = False

DEBUG = False


##################################################
############## EXECUTION VARIABLES ###############
IP = '127.0.0.1'
PORT = 9999
NSETS = 5

world = ''
evalTime = None
evalIdx = 0
evalMaxIdx = 1

experimentStarted = False
cloudLabeled = False
graspingAttempt = False


##################################################
def sendSocketMsg(msg_, shutdownMsg_):
	rospy.loginfo('[MONITOR]...sending msg: ' + msg_)

	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.connect((IP, PORT))
	sock.send(str(msg_))
	sock.close()

	rospy.signal_shutdown(shutdownMsg_)


##################################################
def setsCallback(msg_):
	if msg_.data >= NSETS:
		rospy.loginfo('[MONITOR]...experiment done (%d)', msg_.data)
		sendSocketMsg(str(defs.EXP_DONE), 'Experiment finished')


##################################################
def evalCallback(msg_):
	global evalTime, evalIdx, world

	if evalTime == None:
		evalTime = defs.STAMP_FORMAT.format(datetime.datetime.now())

	prefix = world + '_' + evalTime
	sufix = ''
	if msg_.status == EvaluationStatus.BEFORE_EVAL:
		if not SCREENSHOT_BEFORE:
			return
		sufix = '_BEFORE'

	elif msg_.status == EvaluationStatus.PERFORMING_NEW_EVAL:
		sufix = '_EVAL_' + str(evalIdx)
		evalIdx = evalIdx + 1
		if not SCREENSHOT_EVAL or evalIdx > evalMaxIdx:
			return

	elif msg_.status == EvaluationStatus.AFTER_EVAL:
		sufix = '_AFTER'
		evalTime = None
		evalIdx = 0
		if not SCREENSHOT_AFTER:
			return


	# Take the screenshots
	pattern1 = re.compile('.*Gazebo.*')
	pattern2 = re.compile('.*rviz.*')

	screen = wnck.screen_get_default()
	screen.force_update()
	windows = screen.get_windows()

	os.system('mkdir -p ' + defs.MONITOR_OUTPUT)
	for w in windows:
		if pattern1.match(w.get_name()) and SCREENSHOT_GAZEBO:
			os.system('import -window "' + w.get_name() + '" ' + defs.MONITOR_OUTPUT + prefix + '_gazebo' + sufix +'.jpg')
		elif pattern2.match(w.get_name()) and SCREENSHOT_RVIZ:
			os.system('import -window "' + w.get_name() + '" ' + defs.MONITOR_OUTPUT + prefix + '_rviz' + sufix +'.jpg')


##################################################
def cloudCallback(msg_):
	global experimentStarted, cloudLabeled
	rospy.logdebug('[MONITOR]...labeled cloud received')
	experimentStarted = True
	cloudLabeled = True


##################################################
def graspingCallback(msg_):
	global graspingAttempt
	rospy.logdebug('[MONITOR]...pickup goal received')
	graspingAttempt = True


##################################################
def watchdogStart(event_):
	rospy.logdebug('[MONITOR]...start timer')
	if not experimentStarted:
		sendSocketMsg(str(defs.EXP_START_FAILED), 'Experiment start failed')


##################################################
def watchdogLabeling(event_):
	global cloudLabeled
	rospy.logdebug('[MONITOR]...labeling timer')
	if not cloudLabeled:
		sendSocketMsg(str(defs.EXP_HANGED), 'Timeout cloud labeling')
	else:
		cloudLabeled = False


##################################################
def watchdogGrasping(event_):
	global graspingAttempt
	rospy.logdebug('[MONITOR]...grasping timer')
	if not graspingAttempt:
		sendSocketMsg(str(defs.EXP_HANGED), 'Timeout grasping attempt')
	else:
		graspingAttempt = False


##################################################
if __name__ == '__main__':
	try:
		if (len(sys.argv) < 5):
			print('NOT ENOUGH ARGUMENTS GIVEN.\n')
			print('   Usage: python experiment_monitor_node.py <ip> <port> <nsets> <world_name>')
			sys.exit(0)


		IP = sys.argv[1]
		PORT = int(sys.argv[2])
		NSETS = int(sys.argv[3])
		world = sys.argv[4]


		logLevel = rospy.INFO
		if DEBUG:
			logLevel = rospy.DEBUG
		rospy.init_node('experiment_monitor', anonymous=False, log_level=logLevel)


		# set a watchdog for faulty starts
		rospy.Timer(rospy.Duration(120), watchdogStart, oneshot=True)
		rospy.Timer(rospy.Duration(1200), watchdogLabeling, oneshot=False)
		rospy.Timer(rospy.Duration(600), watchdogGrasping, oneshot=False)


		rospy.Subscriber('/pr2_grasping/processed_sets', Int32, setsCallback)
		rospy.Subscriber('/pr2_grasping/evaluation_status', EvaluationStatus, evalCallback)
		rospy.Subscriber('/pr2_grasping/labeled_cloud', PointCloud2, cloudCallback)
		rospy.Subscriber('/pickup/goal', PickupActionGoal, graspingCallback)

		rospy.spin()

	except rospy.ROSInterruptException as e:
		print('Node interrupted: ' + str(e))
		pass