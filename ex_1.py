import numpy as np
import sys
import scipy.io.wavfile
sample, centroids = sys.argv[1], sys.argv[2]
fs, y = scipy.io.wavfile.read(sample)
x = np.array(y.copy())
centroids = np.loadtxt(centroids)
savedCentroids = centroids.copy()
firstCent = centroids[0]
outputFile = open("output.txt", "w")
# going for 30 epochs at most , or we will stop at convergence.
for epoch in range(30):
    mappedPoints = np.zeros([len(centroids), 2])
    clusterSize = np.zeros([len(centroids)])
    for point in x:
        minDistance = ((firstCent[0] - point[0])**2 + (firstCent[1] - point[1])**2)
        centroidNumber = 0
        chosenCentroid = 0
        for centroid in centroids:
            # calculating distance
            tempDistance = ((centroid[0] - point[0])**2 + (centroid[1] - point[1])**2)
            # looking for the centroid with the smallest distance to the given point
            if tempDistance < minDistance:
                minDistance = tempDistance
                chosenCentroid = centroidNumber
            centroidNumber += 1
        mappedPoints[chosenCentroid] += point
        clusterSize[chosenCentroid] += 1
    # updating the centroids location to be the average of their matching points
    for i in range(len(mappedPoints)):
        if clusterSize[i] == 0:  # avoiding empty cluster exception
            continue
        mappedPoint = mappedPoints[i]
        newX = round(mappedPoint[0] / clusterSize[i])
        newY = round(mappedPoint[1] / clusterSize[i])
        centroids[i][0] = newX
        centroids[i][1] = newY
    # checking for convergence
    if (savedCentroids == centroids).all():
        outputFile.write(f"[iter {epoch}]:{','.join([str(centroid) for centroid in centroids])}")
        outputFile.write("\n")
        break
    else:
        savedCentroids = centroids.copy()
    outputFile.write(f"[iter {epoch}]:{','.join([str(centroid) for centroid in centroids])}")
    outputFile.write("\n")
outputFile.close()
