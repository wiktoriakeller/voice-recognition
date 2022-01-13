import numpy as np
import os
import sys
import warnings
from scipy import signal
from scipy.io import wavfile
import scipy.fft

def recognizeVoice(path, file):
    fullpath = path + file

    try:
        samplerate, data = wavfile.read(fullpath)
    except:
        return ('No such a file', -1)

    if len(data.shape) == 2:
        data = data.sum(axis=1) / 2.0

    dataLength = len(data)
    start = int(0.28 * dataLength)
    end = dataLength - start
    data = data[start:end]
    dataLength = len(data)
    
    beta = 4
    kaiserWindow = signal.windows.kaiser(dataLength, beta)

    data = kaiserWindow * data
    spectrum = abs(scipy.fft.rfft(data))    

    downsampledCount = 5
    resultSignal = spectrum.copy()

    for i in range(2, downsampledCount):
        downsampled = signal.decimate(spectrum, i)
        length = len(downsampled)
        resultSignal = resultSignal[:length] * downsampled[:length]
    
    cutoff = 70
    maxFreq = ((np.argmax(resultSignal[cutoff:]) + cutoff) * samplerate) / dataLength

    if maxFreq >= 165:
        return ('K', maxFreq)

    return ('M', maxFreq)

def statistics(path):
    womanVoicesCount = 0
    manVoicesCount = 0

    resultsDict = {}
    resultsDict['M'] = 0
    resultsDict['K'] = 0
    
    for file in os.listdir(path):
        voiceInFile = file.split('_')[1].split('.')[0]
        if voiceInFile == 'M':
            manVoicesCount += 1
        else:
            womanVoicesCount += 1
        
        result = recognizeVoice(path, file)
        if result[0] == voiceInFile:
            resultsDict[voiceInFile] += 1
            print(file + ' - Correct ' + ' result: ' + str(result[0]) + ' frequency: ' + str(result[1]))
        else:
            print(file + ' - Incorrect ' + ' result: ' + str(result[0]) + ' frequency: ' + str(result[1]))

    totalCorrect = round((resultsDict['M'] + resultsDict['K']) / (womanVoicesCount + manVoicesCount), 4) * 100
    manCorrect = round(resultsDict['M'] / manVoicesCount, 4) * 100
    womanCorrect = round(resultsDict['K'] / womanVoicesCount, 4) * 100

    print('Total correct: ' + str(resultsDict['M'] + resultsDict['K']) + '/' + str(manVoicesCount + womanVoicesCount))
    print('Total correct percentage: ' + str(totalCorrect))

    print('Correct woman voice recognition: ' + str(resultsDict['K']) + '/' + str(womanVoicesCount))
    print('Correct woman voice recognition percantage: ' + str(womanCorrect))

    print('Correct man voice recognition: ' + str(resultsDict['M']) + '/' + str(manVoicesCount))
    print('Correct man voice recognition percentage: ' + str(manCorrect))
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    mainDir = os.path.dirname(os.path.realpath(__file__))
    trainPath = mainDir + '\\train\\'
    #statistics(trainPath)

    if len(sys.argv) < 2:
        print('File name has not been provided')
    else:
        fileName = sys.argv[1]
        result = recognizeVoice(trainPath, fileName)
        print(result[0])
