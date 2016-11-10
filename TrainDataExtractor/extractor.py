'''
@author: rodrigo
2016
'''
import yaml
import os
import sys
import csv
from collections import OrderedDict


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
	for f in os.listdir(dir_):
		element = dir_ + f
		if (os.path.isdir(element)):
			print('...processing ' + element)
			traverseDirectory(element + '/', fields_, writer_)
		else:
			try:
				# generate the output CSV file
				with open(element, 'r') as ff:

					# generate dictionary to extract the data
					extracted = dict((e, None) for e in fields_.keys())

					# perform the extraction
					data = yaml.load(ff)
					for key in fields_:
						seq = fields_[key]

						# read from YAML only the existing keys
						if seq[0] in data:
							node = data[seq[0]]
							for i in range(1, len(seq)):
								node = node[seq[i]]
							extracted[key] = node

					extracted['experiment'] = f
					extracted['set'] = dir_.strip('/').split('/')[-1]

					writer_.writerow(extracted)


			except IOError as e:
				print('Unable to file ' + f)


##################################################
if __name__ == '__main__':

	if (len(sys.argv) < 2):
		print('NOT ENOUGH ARGUMENTS GIVEN.\n')
		print('   Usage: python extractor.py <target_dir>\n\n')
		sys.exit(0)

	# get the fields to extract
	fields = getFields()

	try:
		# generate the output CSV file
		with open('extracted_data.csv', 'wb') as csvFile:
			wr = csv.DictWriter(csvFile, fields.keys())
			wr.writeheader()
			traverseDirectory(sys.argv[1], fields, wr)

	except IOError as e:
		print('Unable to create output file')

	print('Extraction finished')