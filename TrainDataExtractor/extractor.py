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
import matplotlib.pyplot as plt


##################################################
################## APP'S CONFIG ##################
LOG_LEVEL = logging.INFO
USE_ANGLE = True
USE_AUTO_TRAIN = False
TEST_SVM_PREDICTIONS = True
OUTPUT_DIR = './output/'

##################################################
logger = None
nclusters = 7
nsplit = 4


def gaussianKernel(alpha_, y_):
	return (numpy.exp(-0.5 * alpha_ * alpha_) / numpy.sqrt(2 * numpy.pi)) * y_

def cosineKernel(alpha_, y_):
	return (numpy.pi * numpy.cos(numpy.pi * alpha_ / 2) / 4) * y_


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
def extractTrainData(finput_, foutput_, fields_):
	try:
		# generate a CSV file and extract training data
		logger.debug('Opening output file')
		with open(foutput_, 'wb') as csvFile:
			wr = csv.DictWriter(csvFile, fields_.keys())
			wr.writeheader()

			##### DATA DETAILS #####
			# train data is organized as pairs (cluster id, gripper's angle)
			#
			# responses are
			#     successful = False -> 0 -> class 0
			#     successful = True -> 1 -> class 1
			#
			logger.info('...extracting data')
			return traverseDirectory(finput_, fields_, wr)

	except IOError as e:
		logger.info('Unable to create output file')


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

						##### RESPONSE ORGANIZATION #####
						# successful = False -> 0 -> class 0
						# successful = True -> 1 -> class 1
						response = response + [int(extracted['successful'])] 

					logger.debug('\t...extracted: ' + str(tmp))

			except IOError as e:
				logger.info('Unable to file ' + f)

	result = [train, response]
	logger.debug('...returning: ' + str(result))
	return result


##################################################
def testPredictions(svm_):
	logger.info('Testing SVM predictions')

	step = numpy.pi / nsplit
	for label in range(nclusters):
		for angle in range(nsplit):
			tmp = numpy.array([[label, angle * step]], dtype = numpy.float32)
			res = svm.predict(tmp)
			distance = svm.predict(tmp, True)
			logger.info('...prediction (%d, %.2f): % 3.3f / %s', tmp[0][0], tmp[0][1], distance, res == 1)


##################################################
def calcDistribution(svm_):
	logger.info('Testing SVM predictions')

	# result = numpy.array()

	step = numpy.pi / nsplit
	angles = numpy.arange(-numpy.pi, numpy.pi, step)
	for label in range(nclusters):
		logger.info('...evaluating label ' + str(label))

		# generate the results for the current cluster
		results = numpy.array([])
		for angle in angles:
			d = numpy.array([[label, angle]], dtype = numpy.float32)
			p = svm.predict(d)
			results = numpy.append(results, p)

		# evaluate the kernel
		evalStep = numpy.pi / 100
		# evalAngles = numpy.arange(0, numpy.pi, evalStep)
		evalAngles = numpy.arange(-numpy.pi, numpy.pi, evalStep)

		dist = []
		h = 0.5
		for alpha in evalAngles:
			acc = 0
			# n = nsplit + 1
			n = nsplit
			for j in range(n):
				
				

				delta = (alpha - step * j)
				
				# logger.info('alpha: %.2f - ang: %.2f - delta: %f', alpha, step * j, delta)
				# raw_input()

				# if abs(delta) >= numpy.pi:
				# 	delta = 0
				# 	logger.info('changing!')

				delta = delta / h
				val = gaussianKernel(delta, results[j % nsplit])
				# val = cosineKernel(delta, results[j % nsplit])
				acc = acc + val

			acc = acc / (n * h)
			dist.append(acc)


		tmpAngles = numpy.append(angles, numpy.pi)
		tmpRes = numpy.append(results, results[0])

		fig, ax1 = plt.subplots()
		ax1.plot(evalAngles, dist, 'r--')
		ax1.set_xlabel('Angle')
		ax1.set_ylabel('Value')
		ax2 = ax1.twinx()
		ax2.plot(tmpAngles, tmpRes, 'bs')
		ax2.set_ylabel('Prediction')

		# ax1.set_xlim([0, numpy.pi])
		# ax2.set_xlim([0, numpy.pi])
		# plt.show()


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


	# generate output name
	i = -1
	fname = sys.argv[1].split('/')[i]
	while len(fname) == 0:
		i = i -1
		fname = sys.argv[1].split('/')[i]


	# generate output directory
	os.system('mkdir -p ' + OUTPUT_DIR)


	# get the fields to extract
	logger.debug('Generating fields')
	fields = getFields()


	# retrieve the training data
	train, response = extractTrainData(sys.argv[1], OUTPUT_DIR + fname + '.csv', fields)
	logger.info('Extracted %d items', len(train))


	# train a classifier
	logger.info('Training classifier')
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

	logger.info('Saving model to disk')
	svm.save(OUTPUT_DIR + fname + '.yaml')


	# test classifier predictions
	if TEST_SVM_PREDICTIONS:
		testPredictions(svm)

	# calcDistribution(svm)


	logger.info('Execution finished')
