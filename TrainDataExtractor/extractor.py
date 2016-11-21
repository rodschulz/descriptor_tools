'''
@author: rodrigo
2016
'''
import yaml
import os
import sys
import csv
import numpy
import cv2
import logging
from collections import OrderedDict


##################################################
################## APP'S CONFIG ##################
LOG_LEVEL = logging.INFO
USE_ANGLE = True
USE_AUTO_TRAIN = False
TEST_SVM_PREDICTIONS = True

##################################################
logger = None


##################################################
def getFields():
	fields = OrderedDict()

	fields['object'] = ['target_object']
	fields['completed'] = ['result', 'attempt_completed']
	fields['successful'] = ['result', 'success']
	fields['errCode'] = ['result', 'pick_error_code']
	fields['cluster'] = ['cluster_label']
	fields['angle'] = ['orientation', 'angle']
	fields['splits'] = ['orientation', 'split_number']
	fields['graspId'] = ['grasp', 'id']
	fields['experiment'] = ['']
	fields['set'] = ['']

	return fields


##################################################
def traverseDirectory(dir_, fields_, writer_):
	train = []
	response = []

	# iterate over files in the directory
	for f in os.listdir(dir_):
		element = dir_ + f

		if (os.path.isdir(element)):
			logger.info('...traversing ' + f)
			t, r = traverseDirectory(element + '/', fields_, writer_)

			if len(t) > 0:
				train = train + t
				response = response + r

		else:
			# process only YAML files, skip the rest
			if not f.split('.')[-1].lower() == 'yaml':
				continue

			logger.info('...extracting ' + f)
			try:
				# generate the output CSV file
				with open(element, 'r') as ff:

					# generate dictionary to extract the data
					extracted = dict((e, None) for e in fields_.keys())

					# perform the extraction
					fileData = yaml.load(ff)
					for key in fields_:
						seq = fields_[key]

						# read from YAML only the existing keys
						if seq[0] in fileData:
							node = fileData[seq[0]]
							for i in range(1, len(seq)):
								node = node[seq[i]]
							extracted[key] = node

					extracted['experiment'] = f
					extracted['set'] = dir_.strip('/').split('/')[-1]

					# write data to CSV file
					writer_.writerow(extracted)

					# return the data for training
					tmp = []
					if extracted['completed']:
						
						if USE_ANGLE:
							tmp = [extracted['cluster'], extracted['angle']]
						else:
							angle = extracted['angle']
							splits = extracted['splits']
							tmp = [extracted['cluster'], int(round(angle * splits / numpy.pi))]

						train = train + [tmp]
						response = response + [int(extracted['successful'])]

					logger.debug('\t...extracted: ' + str(tmp))

			except IOError as e:
				logger.info('Unable to file ' + f)

	result = [train, response]
	logger.debug('...returning: ' + str(result))
	return result


##################################################
if __name__ == '__main__':
	if (len(sys.argv) < 2):
		print('NOT ENOUGH ARGUMENTS GIVEN.\n')
		print('   Usage: python extractor.py <target_dir>\n\n')
		sys.exit(0)

	# setup logging
	formatter = logging.Formatter('[%(levelname)-5s] %(message)s')
	logger = logging.getLogger('EXTRACTOR')
	logger.setLevel(LOG_LEVEL)
	ch = logging.StreamHandler()
	ch.setFormatter(formatter)
	logger.addHandler(ch)


	# get the fields to extract
	logger.debug('Generating fields')
	fields = getFields()


	try:
		# generate a CSV file and extract training data
		logger.debug('Opening output file')
		with open('extracted_data.csv', 'wb') as csvFile:
			wr = csv.DictWriter(csvFile, fields.keys())
			wr.writeheader()
			train, response = traverseDirectory(sys.argv[1], fields, wr)
			logger.info('Extracted %d items', len(train))

		logger.info('Extraction finished')

	except IOError as e:
		logger.info('Unable to create output file')


	# train a classificator
	logger.info('Training classificator')

	tt = numpy.array(train, dtype = numpy.float32)
	rr = numpy.array(response, dtype = numpy.float32)

	svm = cv2.SVM()
	svmParams = dict(kernel_type = cv2.SVM_RBF, svm_type = cv2.SVM_C_SVC, term_crit=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.0000000001))

	if USE_AUTO_TRAIN:
		logger.info('...using AUTO_TRAIN')
		svm.train_auto(tt, rr, None, None, params=svmParams, k_fold=10)
	else:
		svmParams['C'] = 2.78
		svm.train(tt, rr, params=svmParams)

	svm.save('svm.yaml')

	if TEST_SVM_PREDICTIONS:
		logger.info('Testing SVM predictions')
		nsplit = 4
		step = numpy.pi / nsplit

		numpy.set_printoptions(precision=1)
		for label in range(7):
			for angle in range(nsplit):
				tmp = numpy.array([[label, angle * step]], dtype = numpy.float32)
				res = svm.predict(tmp)
				logger.info('...predicting (%d, %.2f): ' + str(res == 1), tmp[0][0], tmp[0][1])
