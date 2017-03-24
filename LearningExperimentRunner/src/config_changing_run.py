'''
@author: rodrigo
2016
'''
import time
import ruamel.yaml
from ruamel.yaml.util import load_yaml_guess_indent
import subprocess


CONFIG_DIR = '/home/rodrigo/Documents/catkin_ws/src/grasping/config/'
LOCAL_DIR = './config/'
CLASSIFIER_DIR = LOCAL_DIR + 'classifiers/'
CONFIG_FILE = CONFIG_DIR + 'config.yaml'


# angles = [4, 5, 6]
angles = [5, 6]
classifiers = ['DCH_72_split4_beer_drill_svm_auto.yaml', 'DCH_72_split5_beer_drill_svm_auto.yaml', 'DCH_72_split45_beer_drill_svm_auto.yaml']


for c in classifiers:
	for a in angles:
		config, ind, bsi = load_yaml_guess_indent(open(CONFIG_FILE))
		
		config['grasper']['angleSplits'] = a
		config['grasper']['predictions']['classifier'] = CLASSIFIER_DIR + c

		ruamel.yaml.round_trip_dump(config, open(CONFIG_FILE, 'w'), indent=ind, block_seq_indent=bsi)

		CMD = 'python src/run.py ~/Documents/catkin_ws/ world_set_1 true'
		process = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE)
		process.wait()

		time.sleep(10)

print 'FINISHED'