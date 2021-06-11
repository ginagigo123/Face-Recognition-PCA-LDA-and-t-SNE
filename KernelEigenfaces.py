# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:59:19 2021

@author: ginag
"""
import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt

# calculate the distance
import scipy.spatial

K = [1, 3, 5, 7]
subjectNum = 15

#RESIZE = (231, 195)
RESIZE = (60, 50)

def readPGM(file):
    with open(file, "rb") as f:
        # f.readline() # P5
        # f.readline() # Comment line
        # width, height = [int(i) for i in f.readline().split()]
        img = Image.open(f)
        img = img.resize(RESIZE, Image.ANTIALIAS)        
        imgArray = np.array(img)
        return imgArray.flatten()

def readFile(filePath):
    filename = []
    label = []
    img = []
    for file in os.listdir(filePath):
        filename.append(file)
        label.append(file[:file.find(".")])
        img.append( readPGM( filePath + "/" + file ) )
    return np.array(filename), np.array(label), np.array(img)

def PCA(train, K):
    # average face 
    avgFace = np.mean(train, axis = 1)
    
    # transformation - compute the covariance (every image - average face)
    X = (train.T - avgFace).T
    cov_ = X.T @ X
    
    # do the eigen decomposition, eigh => eigenValue would be ascending
    eigenValue, eigenVector = np.linalg.eigh(cov_)
    
    # transformation - train @ eigenvector 
    eigenVector = X @ eigenVector
    
    # normalize the eigenVector => ||w|| = 1
    for i in range(eigenVector.shape[1]):
        eigenVector[ :, i] = eigenVector[ :, i] / np.linalg.norm( eigenVector[:, i] )
        
    # select the largest K eigenVector
    W = eigenVector[: , -K:]
    return avgFace, W

def LDA(train, K):
    # how many cluster
    imgNum = train.shape[1] // subjectNum
    mean = np.zeros(( subjectNum, train.shape[0]))
    for i in range(subjectNum):
        mean[i] = np.mean(train[:, i * imgNum : (i+1) * imgNum], axis=1)
    
    overallMean = np.mean(train, axis=1)
    
    # compute the between-class scatter and within-class scatter
    Sw = np.zeros((train.shape[0], train.shape[0]))
    Sb = np.zeros((train.shape[0], train.shape[0]))
    
    for index, value in enumerate(np.unique(label)):
        xi = train[:, np.where(label == value)[0]].T
        dist1 = xi - mean[index]
        Sw += scipy.spatial.distance.cdist(dist1.T, dist1.T, 'euclidean')
        
        dist2 = (mean[index] - overallMean).reshape(1, -1)
        Sb += len(np.where(label == value)[0]) * scipy.spatial.distance.cdist(dist2.T, dist2.T, 'euclidean')
    
    # compute for the eigenVector
    fisherValue, fisherVector = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    
    #select the largest K fisherVector
    index = np.argsort(fisherValue)[::-1]
    W = fisherVector[:, index[: K]].real
    return overallMean, W
    

def predictFaceRecong(train_proj, label, test_proj, testLabel):
    dist = scipy.spatial.distance.cdist(test_proj.T, train_proj.T, 'euclidean')
    
    # different k -> k for nearest neighbor
    for k in K:
        truePredict = 0
        for i in range(dist.shape[0]):
            row = dist[i, :]
            sortIndex = np.argsort(row)[: k]
            neighbor = label[sortIndex]
            face, count = np.unique(neighbor, return_counts=True)
            predict = face[np.argmax(count)]
            if predict == testLabel[i]:
                truePredict += 1
            #print("predict : ", predict, "// True : ", testLabel[i])
        print(f'K neighbors={k}, Accuracy: {truePredict / test_proj.shape[1]:>.3f} ({truePredict}/{test_proj.shape[1]})')
        
def linearKernel(X):
    return X @ X.T

def polynomialKernel(X, gamma=1e-3, coef=0.7, degree=3):
    return np.power(gamma * (X @ X.T) + coef, degree)

def rbfKernel(X, gamma=1e-7):
    return np.exp(-gamma * scipy.spatial.distance.cdist(X, X, 'sqeuclidean'))

def KernelPCA(train, K, method):
    # compute kernel
    kernel = method(train)
    one = np.full((kernel.shape[0], kernel.shape[0]), 1 / train.shape[1])
    # make the data to be centered already
    centeredKernel = kernel - one @ kernel - kernel @ one + one @ kernel @ one
    eigenValue, eigenVector = np.linalg.eigh(centeredKernel)
    
    #index = np.argsort(eigenValue)
    # already sorted due to the implementation of eigh
    W = eigenVector[:, -K:]
    return W
    
def KernelLDA(train, K, method):
    _kernel = method(train.T)
    
    # compute the between-class scatter and within-class scatter
    Z = np.full((_kernel.shape[0], _kernel.shape[0]), 1 / _kernel.shape[0])
    Sb = _kernel @ Z @ _kernel
    Sw = _kernel @ _kernel
    
    # compute for the eigenVector
    fisherValue, fisherVector = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    
    #select the largest K fisherVector
    index = np.argsort(fisherValue)[::-1]
    W = fisherVector[:, index[: K]].real
    return W

def drawEigenFace(eigenFace, K, title):
    # for ploting multiple figures
    fig, ax = plt.subplots(5, 5, figsize=(8, 8), squeeze=False)
    fig.tight_layout(pad = 3.0)
    
    fig.suptitle(title, fontsize=16)
    for i in range(K):
        # original image
        r, c = i // 5, i % 5
        ax[r][c].imshow(eigenFace[:, i].reshape(RESIZE[1], RESIZE[0]), cmap="gray")
        ax[r][c].set_title(title[3:] + "_" + str(i))
    
    plt.subplots_adjust(top=0.9)
    plt.savefig(title)

def drawReconstructFace(avgFace, eigenFace, train, filename, randIndex, title):
    # for ploting multiple figures
    fig, ax = plt.subplots(4, 5, figsize=(12, 8), squeeze=False)
    fig.tight_layout(pad = 3.0)
    fig.suptitle( title + ' Face Reconstruct', fontsize=16)
    
    for index, value in enumerate(randIndex):
        img = train[:, value]
        projection = eigenFace.T @ (img - avgFace)
        reFace = eigenFace @ projection + avgFace
        
        r, c = index // 5 * 2, index % 5
        ax[r][c].imshow(train[:, value].reshape(RESIZE[1], RESIZE[0]), cmap="gray")
        ax[r][c].set_title("original")
        
        ax[r+1][c].imshow(reFace.reshape(RESIZE[1], RESIZE[0]), cmap="gray")
        ax[r+1][c].set_title(filename[value])
    plt.subplots_adjust(top=0.9)        
    plt.savefig(title + "_Face_Reconstruct.png")


if __name__ == '__main__':
    
    filename, label, train = readFile("./Yale_Face_Database/Training")
    train = train.T
    
    testFilename, testLabel, test = readFile("./Yale_Face_Database/Testing")
    
    #os.mkdir("LDA")
    
    print("-" * 10)
    choice = input("1. Face Reconstruction:")
    
    # show first 25 eigenFaces and fisherFaces and pick random 10 images to recontruct
    if choice == "1":
        # PCA
        avgFace, eigenFace = PCA(train, 25)
        randIndex = np.random.choice(train.shape[1], 10, replace=False)
        
        # ploting
        drawEigenFace(eigenFace, 25, "PCA/PCA EigenSpace")
        drawReconstructFace(avgFace, eigenFace, train, filename, randIndex, "PCA/PCA")
        
        # # LDA
        avgFace, fisherFace = LDA(train, 25)
        randIndex = np.random.choice(train.shape[1], 10, replace=False)
        
        # ploting
        drawEigenFace(fisherFace, 25, "LDA/LDA fisherSpace1")
        drawReconstructFace(avgFace, fisherFace, train, filename, randIndex, "LDA/LDA1")
    
    # face recognition, predict the test (knn) and output performance
    elif choice == "2":
        # PCA
        avgFace, eigenFace = PCA(train, 25)
        train_proj = eigenFace.T @ (train.T - avgFace).T 
        test_proj = eigenFace.T @ (test - avgFace).T
        print("-"*10, "PCA", "-"*10)
        predictFaceRecong(train_proj, label, test_proj, testLabel)
        
        # LDA
        avgFace, fisherFace = LDA(train, 25)
        train_proj = fisherFace.T @ (train.T - avgFace).T 
        test_proj = fisherFace.T @ (test - avgFace).T
        print("-"*10, "LDA", "-"*10)
        predictFaceRecong(train_proj, label, test_proj, testLabel)
        
    # kernel PCA and LDA, face recognition and compute the performance
    else:
        # Kernel PCA and LDA
        kernel = [linearKernel, polynomialKernel, rbfKernel]
        for method in kernel:
            # PCA
            eigenFace = KernelPCA(train, 25, method)
            train_proj = eigenFace.T @ (train.T - avgFace).T 
            test_proj = eigenFace.T @ (test - avgFace).T
            print("-"*10, "Kernel PCA - ", str(method), "-"*10)
            predictFaceRecong(train_proj, label, test_proj, testLabel)
            
        for method in kernel:
            # LDA
            fisherFace  = KernelLDA(train, 25, method)
            train_proj = eigenFace.T @ (train.T - avgFace).T 
            test_proj = eigenFace.T @ (test - avgFace).T
            print("-"*10, "Kernel LDA - ", str(method), "-"*10)
            predictFaceRecong(train_proj, label, test_proj, testLabel)
            

        
        


"""
#plt.imshow((avgFace).reshape(50, 60), cmap="gray")

imgArray = readPGM("./Yale_Face_Database/Training/subject01.centerlight.pgm")
test = readPGM("./Yale_Face_Database/Training/subject01.happy.pgm")


X = np.full((231, 231), 255)

X[ : imgArray.shape[0], : imgArray.shape[1]] = imgArray

meanX = np.average(X)
S = 1 / imgArray.shape[0] * (X - meanX) @ (X - meanX).T

eigenValue, eigenVector = np.linalg.eig(S)

k = eigenVector[:, :25]
z = X @ k @ k.T

z = z.real
"""
#plt.imshow((train[0] - mean).reshape(231, 195), cmap="gray")