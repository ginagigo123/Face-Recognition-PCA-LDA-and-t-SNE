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
#RESIZE = (100, 100)
RESIZE = (60, 50)

def readPGM(file):
    with open(file, "rb") as f:
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
        Sw += dist1.T @ dist1
        
        dist2 = (mean[index] - overallMean).reshape(1, -1)
        Sb += len(np.where(label == value)[0]) * dist2.T @ dist2
    
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
        
def linearKernel(X, Y):
    return X @ Y.T

def polynomialKernel(X, Y, gamma=1e-2, coef=0.1, degree=2):
    return np.power(gamma * (X @ Y.T) + coef, degree)

def rbfKernel(X, Y, gamma=1e-8): 
    return np.exp(-gamma * scipy.spatial.distance.cdist(X, Y, 'sqeuclidean'))

def KernelPCA(train, K, method):
    # compute kernel
    kernel = method(train.T, train.T)
    eigenValue, eigenVector = np.linalg.eigh(kernel)
    
    index = np.argsort(eigenValue)[::-1]
    # already sorted due to the implementation of eigh
    W = eigenVector[:, index[:K]].real
    return W, kernel
    
def KernelLDA(train, K, method):
    _kernel = method(train.T, train.T)
    
    # compute the between-class scatter and within-class scatter
    Z = np.full((_kernel.shape[0], _kernel.shape[0]), 1 / _kernel.shape[0])
    Sb = _kernel @ Z @ _kernel
    Sw = _kernel @ _kernel
    
    # compute for the eigenVector
    fisherValue, fisherVector = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    
    #select the largest K fisherVector
    index = np.argsort(fisherValue)[::-1]
    W = fisherVector[:, index[: K]].real
    return W, _kernel

def drawEigenFace(eigenFace, K, title):
    # for ploting multiple figures
    fig, ax = plt.subplots(5, 5, figsize=(8, 8), squeeze=False)
    fig.tight_layout(pad = 3.0)
    
    fig.suptitle(title[4:], fontsize=16)
    for i in range(K):
        # original image
        r, c = i // 5, i % 5
        ax[r][c].imshow(eigenFace[:, i].reshape(RESIZE[1], RESIZE[0]), cmap="gray")
        ax[r][c].set_title("Eigenface_" + str(i))
    
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
        if title[:3] == "LDA":
            reFace = eigenFace @ projection
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
    print("1.PCA/LDA Face Reconstruction, 2. PCA/LDA Face Recogntion, 3.Kernel PCA/LDA Face Recognition:")
    choice = input()
    
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
        drawEigenFace(fisherFace, 25, "LDA/LDA fisherSpace")
        drawReconstructFace(avgFace, fisherFace, train, filename, randIndex, "LDA/LDA")
    
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
        
        # Before doing kernel -> make the data centered!!!
        avgFace = np.mean(train, axis=1)
        train = (train.T - avgFace).T
        test = test - avgFace
        
        for method in kernel:
            # PCA
            eigenFace, GramMatrix = KernelPCA(train, 25, method)
            train_proj = GramMatrix @ eigenFace
            
            testGramMatrix = method(test, train.T)
            test_proj = testGramMatrix @ eigenFace
            print("-"*10, "Kernel PCA - ", str(method), "-"*10)
            predictFaceRecong(train_proj.T, label, test_proj.T, testLabel)
            
        for method in kernel:
            # LDA
            fisherFace, GramMatrix  = KernelLDA(train, 25, method)
            train_proj = (GramMatrix) @ fisherFace
            
            testGramMatrix = method(test, train.T)
            test_proj = testGramMatrix @ fisherFace
            print("-"*10, "Kernel LDA - ", str(method), "-"*10)
            predictFaceRecong(train_proj.T, label, test_proj.T, testLabel)
            

        
        

#plt.imshow((train[0] - mean).reshape(231, 195), cmap="gray")

#1 /(RESIZE[0] * RESIZE[1] * 255 * 255)
    # dist = np.sum(X ** 2, axis=1).reshape(-1, 1) \
    #     + np.sum(Y **2, axis=1) \
    #         - 2 * X @ Y.T
    # return np.exp(-gamma *  dist)
    #print(scipy.spatial.distance.cdist(X, Y, 'sqeuclidean'))
