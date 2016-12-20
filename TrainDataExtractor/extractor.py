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
USE_VECTOR = True
SHOW_MATRIX = True
SHOW_STATS = True
OUTPUT_DIR = './output/'

##################################################
logger = None
nsplit = None
nclusters = 7


##################################################
def getBaseName(inputdir_, outdir_):
	i = -1
	fname = inputdir_.split('/')[i]
	while len(fname) == 0:
		i = i -1
		fname = inputdir_.split('/')[i]

	return outdir_ + fname


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
def genRowCSV(dataFile_, fields_, experimentName_, setName_, csvwriter_):
	# retrieve data
	data = dict((e, None) for e in fields_.keys())
	for key in fields_:
		seq = fields_[key]

		# read from YAML only the existing keys
		if seq[0] in dataFile_:
			node = dataFile_[seq[0]]
			for i in range(1, len(seq)):
				node = node[seq[i]]
			data[key] = node

	# fill additional info
	data['experiment'] = experimentName_
	data['set'] = setName_

	# write row to csv file
	if csvwriter_ != None:
		csvwriter_.writerow(data)

	return data


##################################################
def checkSplits(current_, new_):
	if current_ == -1:
		return new_
	elif current_ != new_:
		logger.warn('Multiple splits found: %d / %d', int(current_), int(new_))

	return current_


##################################################
def traverseDirectory(input_, fields_, csvwriter_ = None):
	data = []
	response = []
	nfiles = 0
	nsplits = -1

	# iterate over files in the directory
	setName = input_.strip('/').split('/')[-1]
	for f in os.listdir(input_):
		element = input_ + f

		if (os.path.isdir(element)):
			logger.debug('...traversing ' + f)
			dat, resp, nf, ns = traverseDirectory(element + '/', fields_, csvwriter_)

			if len(dat) > 0:
				data = data + dat
				response = response + resp
				nfiles = nfiles + n
				nsplits = checkSplits(nsplits, ns)

		else:
			# process only YAML files, skip the rest
			if not f.split('.')[-1].lower() == 'yaml':
				continue

			logger.debug('...extracting ' + f)
			nfiles = nfiles + 1

			try:
				# open the results file
				with open(element, 'r') as yamlFile:
					filedata = yaml.load(yamlFile)
					dat = genRowCSV(filedata, fields_, f, setName, csvwriter_)
					nsplits = checkSplits(nsplits, dat['splits'])

					tmp = []
					if dat['completed']:
						if USE_VECTOR:
							tmp = filedata['descriptor']['data'] + [dat['angle']]
						else:
							tmp = [dat['cluster'], dat['angle']]
						data = data + [tmp]

						##### RESPONSE ORGANIZATION #####
						# successful = False -> 0 -> class 0
						# successful = True -> 1 -> class 1
						response = response + [int(dat['successful'])] 

					logger.debug('\t...extracted: ' + str(tmp))

			except IOError as e:
				logger.info('Unable to file ' + f)

	return [data, response, nfiles, nsplits]


##################################################
def extractToCSV(inputdir_, foutput_, csvfields_):
	try:
		# generate a CSV file and extract training data
		logger.debug('Opening output file')
		with open(foutput_, 'wb') as csvfile:
			wr = csv.DictWriter(csvfile, csvfields_.keys(), extrasaction='ignore')
			wr.writeheader()

			##### DATA DETAILS #####
			# train data is organized as pairs (cluster id, gripper's angle)
			#
			# responses are
			#     successful = False -> 0 -> class 0
			#     successful = True -> 1 -> class 1
			#
			logger.info('...extracting data')
			return traverseDirectory(inputdir_, csvfields_, wr)

	except IOError as e:
		logger.info('Unable to create output file')


##################################################
def trainSVM(input_, response_):
	logger.info('...training SVM')

	svmParams = dict(kernel_type = cv2.SVM_RBF, svm_type = cv2.SVM_C_SVC, term_crit=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.0000000001))

	svm_auto = cv2.SVM()
	svm_auto.train_auto(input_, response_, None, None, params=svmParams, k_fold=7)

	svmParams['C'] = 100
	svmParams['gamma'] = 2
	svm = cv2.SVM()
	svm.train(input_, response_, params=svmParams)

	return [svm, svm_auto]


##################################################
def trainBoost(input_, response_):
	logger.info('...training boosted classifier')

	boostParams = dict(boost_type = cv2.BOOST_REAL, weak_count = 100, weight_trim_rate = 0.95, cv_folds = 3, max_depth = 1)

	boost = cv2.Boost()
	boost.train(trainData=input_, tflag=cv2.CV_ROW_SAMPLE, responses=response_, params=boostParams, update=False)

	return boost


##################################################
def trainNetwork(input_, response_):
	logger.info('...training neural network')
	# networkParams = dict(train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, term_crit=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.0000000001), bp_dw_scale=0.1, bp_moment_scale=0.1, flags=(cv2.ANN_MLP_UPDATE_WEIGHTS | cv2.ANN_MLP_NO_INPUT_SCALE))
	
	networkParams = dict(train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, term_crit=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5000, 0.000001), bp_dw_scale=0.1, bp_moment_scale=0.1)


	# networkParams = dict(train_method=cv2.ANN_MLP_TRAIN_PARAMS_RPROP, term_crit=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.0000000001), rp_dw0=0.1, rp_dw_plus=1.2, rp_dw_minus=0.5, rp_dw_min=0.1, rp_dw_max=50, flags=(cv2.ANN_MLP_UPDATE_WEIGHTS | cv2.ANN_MLP_NO_INPUT_SCALE))

	# networkParams = dict(train_method=cv2.ANN_MLP_TRAIN_PARAMS_RPROP, term_crit=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.0000000001), rp_dw0=0.1, rp_dw_plus=1.2, rp_dw_minus=0.5, rp_dw_min=0.1, rp_dw_max=50)


	network = cv2.ANN_MLP()
	network.create(layerSizes=numpy.array([len(input_[0]),5,1], dtype=numpy.int32), activateFunc=cv2.ANN_MLP_SIGMOID_SYM, fparam1=1, fparam2=1)

	rr = response_ -1 + response_
	network.train(inputs=input_, outputs=rr, sampleWeights=numpy.ones(len(response_), dtype = numpy.float32), params=networkParams)

	return network


##################################################
def evalClassifier(classifier_, data_, resp_, showMatrix_ = True, showStats_ = True):
	issvm = isinstance(classifier_, type(cv2.SVM()))
	isbst = isinstance(classifier_, type(cv2.Boost()))
	isnet = isinstance(classifier_, type(cv2.ANN_MLP()))

	tp = 0
	tn = 0
	fp = 0
	fn = 0
	total = len(data_)

	for i in range(total):
		sample = numpy.array([data_[i]], dtype = numpy.float32)

		# predict an output
		pred = -1
		if issvm:
			dist = classifier_.predict(sample, returnDFVal=True)
			label = classifier_.predict(sample, returnDFVal=False)
			pred = int(label)
			logger.debug('resp: %d - label: %.1f - dist: % .3f', resp_[i], label, dist)

		elif isbst:
			label = classifier_.predict(sample, returnSum=False)
			votes = classifier_.predict(sample, returnSum=True)
			pred = int(label)
			logger.debug('resp: %d - label: %.1f - votes: % .3f', resp_[i], label, votes)

		elif isnet:
			dummy, out = classifier_.predict(sample)
			pred = int(out < 0)
			logger.debug('resp: %d - out: % .3f', resp_[i], out)

		# accumulate stats
		if pred == 0: # predicted 0
			if resp_[i] == 0: 
				tn = tn + 1
			else:
				fn = fn + 1
		else: # predicted 1
			if resp_[i] == 1: 
				tp = tp + 1
			else:
				fp = fp + 1


	realn = tn + fp
	realp = fn + tp

	# generate a confusion matrix
	matrix = []
	matrix.append('-------------------------------------')
	matrix.append('|         | pred: 0 | pred: 1 | SUM |')
	matrix.append('-------------------------------------')
	matrix.append('| resp: 0 |   {:3n}   |   {:3n}   | {:3n} |'.format(tn, fp, realn))
	matrix.append('| resp: 1 |   {:3n}   |   {:3n}   | {:3n} |'.format(fn, tp, realp))
	matrix.append('-------------------------------------')
	matrix.append('|   SUM   |   {:3n}   |   {:3n}   | {:3n} |'.format(tn + fn, fp + tp, total))
	matrix.append('-------------------------------------')

	stats = []
	if showStats_:
		with numpy.errstate(invalid='ignore'): # temporarily disable warnings
			stats.append('     accuracy:    {: .3f}'.format((tp + tn) / numpy.float64(total)))
			stats.append('     missclass:   {: .3f}'.format((fp + fn) / numpy.float64(total)))
			stats.append('     TPR:         {: .3f}'.format(tp / numpy.float64(realp)))
			stats.append('     FPR:         {: .3f}'.format(fp / numpy.float64(realn)))
			stats.append('     specificity: {: .3f}'.format(tn / numpy.float64(realn)))
			stats.append('     precision:   {: .3f}'.format(tp / numpy.float64(tp + fp)))
			stats.append('     prevalence:  {: .3f}'.format(realp / numpy.float64(total)))


	nmatrix = len(matrix)
	nstats = len(stats)
	rprt = '\n'
	for i in range(nmatrix):
		if showMatrix_:
			rprt = rprt + matrix[i]
		if showStats_ and i < nstats:
			rprt = rprt + stats[i]
		rprt = rprt + '\n'

	logger.info(rprt)


##################################################
def plotData2D(data_, resp_):
	cls0 = data_[resp_ == 0]
	cls1 = data_[resp_ == 1]

	plt.plot(cls0[:,0], cls0[:,1], 'bo')
	plt.plot(cls1[:,0], cls1[:,1], 'rx')
	plt.axis([0, nclusters + 1, -0.5, 3.5])
	plt.show()


##################################################
def testPredictions(svm_, svm_auto_, boost_, network_):
	logger.info('Testing predictions...')
	step = numpy.pi / nsplit

	table = '\n===========================================================================\n'
	table = table + '|  input  |      SVM      |   SVM auto    |    Boosting   |    Network    |\n'
	table = table + '===========================================================================\n'

	for cluster in range(1, nclusters + 1):
		for angle in range(nsplit):
			sample = numpy.array([[cluster, angle * step]], dtype = numpy.float32)

			svmDist = svm_.predict(sample, returnDFVal=True)
			svmLabel = svm_.predict(sample, returnDFVal=False)

			svmAutoDist = svm_auto_.predict(sample, returnDFVal=True)
			svmAutoLabel = svm_auto_.predict(sample, returnDFVal=False)

			boostLabel = boost_.predict(sample, returnSum=False)
			boostVotes = boost_.predict(sample, returnSum=True)

			dummy, networkOut = network_.predict(sample)

			table =  table + '| {:n}, {:.2f} | {: .2f} / {:5s} | {: .2f} / {:5s} | {: .2f} / {:5s} | {: .2f} / {:5s} |\n'.format(cluster, angle * step, svmDist, str(svmLabel == 1), svmAutoDist, str(svmAutoLabel == 1), boostVotes, str(boostLabel == 1), float(networkOut), str(networkOut[0][0] > 0))

		if cluster != nclusters:
			table = table + '---------------------------------------------------------------------------\n'

	table = table + '===========================================================================\n'
	logger.info(table)


##################################################
if __name__ == '__main__':
	if (len(sys.argv) < 2):
		print('NOT ENOUGH ARGUMENTS GIVEN.\n')
		print('   Usage: python extractor.py <train_dir> <val_dir>\n\n')
		sys.exit(0)

	# setup logging
	formatter = logging.Formatter('[%(levelname)-5s] %(message)s')
	logger = logging.getLogger('EXTRACTOR')
	logger.setLevel(LOG_LEVEL)
	ch = logging.StreamHandler()
	ch.setFormatter(formatter)
	logger.addHandler(ch)


	trainDir = sys.argv[1]
	valDir = sys.argv[2]
	baseName = getBaseName(trainDir, OUTPUT_DIR)

	# retrieve the training data
	os.system('mkdir -p ' + OUTPUT_DIR)
	train, tresp, nfiles, nsplit = extractToCSV(trainDir, baseName + '.csv', getFields())
	logger.info('Traversed %d files', nfiles)
	logger.info('\t- %d completed', len(tresp))
	logger.info('\t- %d successful', sum(tresp))
	logger.info('\t- ratio: %f', float(sum(tresp)) / float(len(train)))

	train = numpy.array(train, dtype = numpy.float32)
	tresp = numpy.array(tresp, dtype = numpy.float32)

	# train classifiers
	svm, svm_auto = trainSVM(train, tresp)
	boost = trainBoost(train, tresp)
	network = trainNetwork(train, tresp)


	# evaluate classifiers
	logger.info('*** TRAINING: stats SVM ***')
	evalClassifier(svm, train, tresp, showMatrix_ = SHOW_MATRIX, showStats_ = SHOW_STATS)
	logger.info('*** TRAINING: stats SVM auto ***')
	evalClassifier(svm_auto, train, tresp, showMatrix_ = SHOW_MATRIX, showStats_ = SHOW_STATS)
	logger.info('*** TRAINING: stats Boost ***')
	evalClassifier(boost, train, tresp, showMatrix_ = SHOW_MATRIX, showStats_ = SHOW_STATS)
	logger.info('*** TRAINING: stats NeuralNetwork ***')
	evalClassifier(network, train, tresp, showMatrix_ = SHOW_MATRIX, showStats_ = SHOW_STATS)


	# retrieve validation data
	logger.info('====================================================')
	logger.info('Retrieving validation data')
	val, vresp, nfiles, dummy = traverseDirectory(valDir, getFields())
	logger.info('Traversed %d files', nfiles)
	logger.info('\t- %d completed', len(vresp))
	logger.info('\t- %d successful', sum(vresp))
	logger.info('\t- ratio: %f', float(sum(vresp)) / float(len(val)))

	val = numpy.array(val, dtype = numpy.float32)
	vresp = numpy.array(vresp, dtype = numpy.float32)


	# evaluate using validation data
	logger.info('*** VALIDATION: stats SVM ***')
	evalClassifier(svm, val, vresp, showMatrix_ = SHOW_MATRIX, showStats_ = SHOW_STATS)
	logger.info('*** VALIDATION: stats SVM auto ***')
	evalClassifier(svm_auto, val, vresp, showMatrix_ = SHOW_MATRIX, showStats_ = SHOW_STATS)
	logger.info('*** VALIDATION: stats Boost ***')
	evalClassifier(boost, val, vresp, showMatrix_ = SHOW_MATRIX, showStats_ = SHOW_STATS)
	logger.info('*** VALIDATION: stats NeuralNetwork ***')
	evalClassifier(network, val, vresp, showMatrix_ = SHOW_MATRIX, showStats_ = SHOW_STATS)


	logger.info('Saving classifiers to disk')
	svm_auto.save(baseName + '_svm_auto.yaml')
	svm.save(baseName + '_svm.yaml')
	boost.save(baseName + '_boost.yaml')
	network.save(baseName + '_network.yaml')


	logger.info('Execution finished')
