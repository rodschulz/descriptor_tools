import os
import sys
import subprocess
import shutil
import yaml

dataDirectory = '/home/rodrigo/Documents/RGBD_summary/'
appDirectory = '/home/rodrigo/Documents/repos/descriptor_apps/'
appLocation = 'build/DenseEvaluator'
configLocation = 'config/config_dense_evaluator.yaml' 

##################################################
def getClusterNumber(fileName):
    try:
        with open(fileName, 'r') as configFile:
            config = yaml.load(configFile)
            return config['clustering']['clusterNumber']
             
    except IOError as e:
        print('Error reading ' + fileName + ': ' + str(e) + ')')
        raise(IOError('Cant get configured clusters number'))

##################################################
def getDestination(nclusters):
    experiment = 1
    while(True):
        destination = './clusters_' + str(nclusters) + '_exp_' + str(experiment) + '/'
        if (not os.path.exists(destination)):
            break
        
        experiment = experiment + 1
        
    return destination

##################################################
def moveOutputData(destination, inputLine):
    inputDir = inputLine.split('/')[0]
    inputFile = inputLine.split('/')[1].split('.')[0]
    
    source = appDirectory + 'output/'
    dest = destination + inputDir + '/' + inputFile + '/'
    
    shutil.copytree(source, dest)
    shutil.rmtree(source)
    os.mkdir(source)

##################################################
def main():
    try:
        # get number of clusters to be used in the reduction process
        nclusters = getClusterNumber(appDirectory + configLocation)
        # get destination directory
        destination = getDestination(nclusters)
    
        # read the input and process the data
        FNULL = open(os.devnull, 'w')
        with open(sys.argv[1]) as clouds:
            for line in clouds:
                line = line.replace('\n', '')
                
                # skip empty lines
                if line == '':
                    continue;
                
                print('Processing file: ' + line)
                
                # run the appLocation and wait
                cmd = appDirectory + appLocation + ' ' + dataDirectory + line + ' | tee DenseEvaluator.log; mv DenseEvaluator.log output/' 
                process = subprocess.Popen(cmd, cwd=appDirectory, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
                process.wait()
                
                if process.returncode == 0:
                    print('\t...execution successful, copying results')
                    moveOutputData(destination, line)
                else:
                    print('\t...execution failed')

    except IOError as e:
        print('ERROR: ' + str(e))
    
##################################################
if __name__ == '__main__':
    main()

