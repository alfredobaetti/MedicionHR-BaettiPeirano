import json
import numpy
import numpy as np
import math

def find_nearest(array,value):  
    
        idx = np.searchsorted(array, value, side="left")
        
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1], idx
        
        else:
            return array[idx], idx

def getTS():
    with open("10-01.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    HR = []
    TShr = []
    TSfr = []
    TShraprox = []
    HRaprox = []

    #1392722533137981000 ts hr aprox
    #1392722533137008000 ts frame

    for c in range (len(jsonObject['/FullPackage'])):
        pulserate = jsonObject['/FullPackage'][c]['Value']['pulseRate']
        tsheartrate = jsonObject['/FullPackage'][c]['Timestamp']
        HR.append(pulserate)
        TShr.append(tsheartrate)

    for c in range (len(jsonObject['/Image'])):
        tsframe = jsonObject['/Image'][c]['Timestamp']
        TSfr.append(tsframe)


    for c in TSfr:
        ts, idx = find_nearest(TShr, c)
        TShraprox.append(ts)
        HRaprox.append(HR[idx])
    
    return TSfr, TShraprox, HRaprox



#print(product)
#print(pulserate)
#print(text)    