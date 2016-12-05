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
TEST_PREDICTIONS = True
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
	nfiles = 0

	# iterate over files in the directory
	for f in os.listdir(dir_):
		element = dir_ + f

		if (os.path.isdir(element)):
			logger.info('...traversing ' + f)
			t, r, nf = traverseDirectory(element + '/', fields_, writer_)

			nfiles = nfiles + nf
			if len(t) > 0:
				train = train + t
				response = response + r

		else:
			# process only YAML files, skip the rest
			if not f.split('.')[-1].lower() == 'yaml':
				continue

			nfiles = nfiles + 1

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

	result = [train, response, nfiles]
	logger.debug('...returning: ' + str(result))
	return result


##################################################
def testPredictions(svm_, boost_, network_):
	logger.info('Testing predictions...')
	step = numpy.pi / nsplit

	logger.info('')
	logger.info('===========================================================')
	logger.info('|  input  |      SVM      |    Boosting   |    Network    |')
	logger.info('===========================================================')

	for cluster in range(nclusters):
		for angle in range(nsplit):
			sample = numpy.array([[cluster, angle * step]], dtype = numpy.float32)

			svmDist = svm_.predict(sample, returnDFVal=True)
			svmLabel = svm_.predict(sample, returnDFVal=False)

			boostLabel = boost_.predict(sample, returnSum=False)
			boostVotes = boost_.predict(sample, returnSum=True)

			dummy, networkOut = network_.predict(sample)

			logger.info('| %d, %.2f | % 2.2f / %5s | % 2.2f / %5s | % 2.2f / %5s |', sample[0][0], sample[0][1], svmDist, svmLabel == 1, boostVotes, boostLabel == 1, networkOut, networkOut[0][0] > 0)

	logger.info('===========================================================')
	logger.info('')


##################################################
def trainSVM(input_, response_):
	logger.info('...training SVM')

	svmParams = dict(kernel_type = cv2.SVM_RBF, svm_type = cv2.SVM_C_SVC, term_crit=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.0000000001))

	svm = cv2.SVM()
	if USE_AUTO_TRAIN:
		logger.info('...using AUTO_TRAIN')
		svm.train_auto(input_, response_, None, None, params=svmParams, k_fold=10)
	else:
		svmParams['C'] = 2.78
		svm.train(input_, response_, params=svmParams)

	logger.info('Saving SVM to disk')
	svm.save(OUTPUT_DIR + fname + '_svm.yaml')

	return svm


##################################################
def trainBoost(input_, response_):
	logger.info('...training boosted classifier')

	boostParams = dict(boost_type = cv2.BOOST_REAL, weak_count = 100, weight_trim_rate = 0.95, cv_folds = 10, max_depth = 1)

	boost = cv2.Boost()
	boost.train(trainData=input_, tflag=cv2.CV_ROW_SAMPLE, responses=response_, params=boostParams, update=False)

	logger.info('Saving boosting classifier to disk')
	boost.save(OUTPUT_DIR + fname + '_boost.yaml')

	return boost


##################################################
def trainNetwork(input_, response_):
	logger.info('...training neural network')
	# networkParams = dict(train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, term_crit=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.0000000001), bp_dw_scale=0.1, bp_moment_scale=0.1, flags=(cv2.ANN_MLP_UPDATE_WEIGHTS | cv2.ANN_MLP_NO_INPUT_SCALE))
	networkParams = dict(train_method=cv2.ANN_MLP_TRAIN_PARAMS_RPROP, term_crit=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.0000000001), rp_dw0=0.1, rp_dw_plus=1.2, rp_dw_minus=0.5, rp_dw_min=0.1, rp_dw_max=50, flags=(cv2.ANN_MLP_UPDATE_WEIGHTS | cv2.ANN_MLP_NO_INPUT_SCALE))

	network = cv2.ANN_MLP()
	network.create(layerSizes=numpy.array([2,3,5,3,1], dtype=numpy.int32), activateFunc=cv2.ANN_MLP_SIGMOID_SYM, fparam1=1, fparam2=1)

	rr = response_ -1 + response_
	network.train(inputs=input_, outputs=rr, sampleWeights=numpy.ones(len(response_), dtype = numpy.float32), params=networkParams)

	logger.info('Saving neural network to disk')
	network.save(OUTPUT_DIR + fname + '_network.yaml')

	return network


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
	train, response, nfiles = extractTrainData(sys.argv[1], OUTPUT_DIR + fname + '.csv', fields)
	logger.info('Traversed %d files', nfiles)
	logger.info('\t- %d completed', len(response))
	logger.info('\t- %d successful', sum(response))
	logger.info('\t- ratio: %f', float(sum(response)) / float(len(train)))


	# train a classifier
	logger.info('Preparing data')
	tt = numpy.array(train, dtype = numpy.float32)
	rr = numpy.array(response, dtype = numpy.float32)

	svm = trainSVM(tt, rr)
	boost = trainBoost(tt, rr)
	network = trainNetwork(tt, rr)

	if TEST_PREDICTIONS:
		testPredictions(svm, boost, network)

	# calcDistribution(svm)

	logger.info('Execution finished')
