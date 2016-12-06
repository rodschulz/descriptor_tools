'''
@author: rodrigo
2016
'''

# Format for timestamps
STAMP_FORMAT = '{:%Y-%m-%d_%H%M%S}'

# Output location of the monitor node
MONITOR_OUTPUT = './nodeOutput/'

# Experiment finish states
EXP_DONE = 0
EXP_START_FAILED = 1
EXP_HANGED = 2

# Returns a descriptive string for the given status
def finishString(status_):
	if status_ == EXP_DONE:
		return 'experiment done'
	elif status_ == EXP_START_FAILED:
		return 'experiment start failed'
	elif status_ == EXP_HANGED:
		return 'experiment hanged'
	else:
		return 'unknown status (' + str(status_) + ')'