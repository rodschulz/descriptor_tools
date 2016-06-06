'''
@author: rodrigo
2016
'''
import os
import sys

##################################################
def traverseDirectory(directory, output):
    for f in os.listdir(directory):
        element = directory + f
        if (os.path.isdir(element)):
            traverseDirectory(element + '/', output)
        else:
            print('Found ' + element)
            output.append(element)

##################################################
def main():
    try:
        targetDirectory = sys.argv[1]
        
        # retrieve the files in the folder
        files = []
        traverseDirectory(targetDirectory, files)
        files.sort()
        
        # clean the names from the target directory
        print('Generating output file')
        output = open('./crawled_files', 'w+')
        for f in files:
            output.write(os.path.abspath(f) + '\n')
        
    except IOError as e:
        print('Cant create output file: ' + str(e))

##################################################
if __name__ == '__main__':
    main()
