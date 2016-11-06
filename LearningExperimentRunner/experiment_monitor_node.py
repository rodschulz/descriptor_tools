'''
@author: rodrigo
2016
'''
import sys
import socket
import rospy
from std_msgs.msg import Bool

IP = '127.0.0.1'
PORT = 9999

##################################################
def experimentDone(msg_):
	if msg_.data == 1:
		rospy.loginfo('Experiment done (%d)', msg_.data)

		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((IP, PORT))
		sock.send('experiment done')
		sock.close()

		rospy.signal_shutdown('Finishing experiment')


##################################################
if __name__ == '__main__':
	try:
		IP = sys.argv[1]
		PORT = int(sys.argv[2])

		rospy.init_node('experiment_monitor', anonymous=True)
		rospy.Subscriber('/pr2_grasping/experiment_done', Bool, experimentDone)
		rospy.loginfo("Experiment monitor spinning")
		rospy.spin()

	except rospy.ROSInterruptException as e:
		print('Node interrupted: ' + str(e))
		pass