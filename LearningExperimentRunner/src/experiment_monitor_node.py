'''
@author: rodrigo
2016
'''
import os, sys
import socket
import re, wnck
import datetime, time, sched
import rospy
from threading import Thread, Lock
import definitions as defs
from std_msgs.msg import Int32
from sensor_msgs.msg import PointCloud2
from pr2_grasping.msg import EvaluationStatus
from moveit_msgs.msg import PickupActionGoal
from moveit_msgs.msg import MoveGroupActionGoal
from pr2_controllers_msgs.msg import PointHeadActionGoal
from pr2_controllers_msgs.msg import Pr2GripperCommandActionGoal


##################################################
################# NODE'S CONFIG ##################
SCREENSHOT_BEFORE = True
SCREENSHOT_EVAL = True
SCREENSHOT_AFTER = False

SCREENSHOT_GAZEBO = True
SCREENSHOT_RVIZ = False

DEBUG = True


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


mutex = Lock()
scheduler = sched.scheduler(time.time, time.sleep)
timerStart = None
timerActivity = None


##################################################
def sendSocketMsg(msg_, shutdownMsg_):
	rospy.loginfo('[MONITOR]...sending: ' + msg_ + ' (' + shutdownMsg_ + ')')

	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.connect((IP, PORT))
	sock.send(str(msg_))
	sock.close()

	rospy.signal_shutdown(shutdownMsg_)


##################################################
def resetTimer():
	global mutex, scheduler, timerActivity
	mutex.acquire()
	scheduler.cancel(timerActivity)
	timerActivity = scheduler.enter(300, 1, watchdogExpired, ())
	mutex.release()


##################################################
def setsCallback(msg_):
	if msg_.data >= NSETS:
		scheduler.cancel(timer_)
		rospy.loginfo('[MONITOR]...experiment done (%d)', msg_.data)
		sendSocketMsg(str(defs.EXP_DONE), 'experiment finished')


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
	global timerStart
	if timerStart != None:
		scheduler.cancel(timerStart)
		timerStart = None

	resetTimer()
	rospy.logdebug('[MONITOR]...labeled cloud received')


##################################################
def pickCallback(mgs_):
	resetTimer()
	rospy.logdebug('[MONITOR]...pickup goal received')

##################################################
def moveCallback(mgs_):
	resetTimer()
	rospy.logdebug('[MONITOR]...move goal received')


##################################################
def headCallback(mgs_):
	resetTimer()
	rospy.logdebug('[MONITOR]...head goal received')


##################################################
def gripperCallback(mgs_):
	resetTimer()
	rospy.logdebug('[MONITOR]...gripper goal received')


##################################################
def watchdogStart():
	global timerActivity
	if timerActivity != None:
		scheduler.cancel(timerActivity)
		timerActivity = None

	rospy.logdebug('[MONITOR]...start timer expired')
	sendSocketMsg(str(defs.EXP_START_FAILED), 'experiment start failed')


##################################################
def watchdogExpired():
	global timerStart
	if timerStart != None:
		scheduler.cancel(timerStart)
		timerStart = None

	rospy.loginfo('[MONITOR]...activity timer expired')
	sendSocketMsg(str(defs.EXP_HANGED), 'activity timer expired')


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


		rospy.loginfo('[MONITOR] setting subscriptions')
		rospy.Subscriber('/pr2_grasping/processed_sets', Int32, setsCallback)
		rospy.Subscriber('/pr2_grasping/evaluation_status', EvaluationStatus, evalCallback)
		rospy.Subscriber('/pr2_grasping/labeled_cloud', PointCloud2, cloudCallback)
		
		rospy.Subscriber('/pickup/goal', PickupActionGoal, pickCallback)
		rospy.Subscriber('/move_group/goal', MoveGroupActionGoal, moveCallback)
		rospy.Subscriber('/head_traj_controller/point_head_action/goal', PointHeadActionGoal, headCallback)
		rospy.Subscriber('/r_gripper_controller/gripper_action/goal', Pr2GripperCommandActionGoal, gripperCallback)


		rospy.loginfo('[MONITOR] setting watchdog timers')
		timerStart = scheduler.enter(120, 1, watchdogStart, ())
		timerActivity = scheduler.enter(300, 1, watchdogExpired, ())
		scheduler.run()


		rospy.loginfo('[MONITOR] node spinning')
		rospy.spin()

	except rospy.ROSInterruptException as e:
		print('Node interrupted: ' + str(e))
		pass