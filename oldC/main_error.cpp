//
//  main.cpp
//  Babel_SGD
//
//  Created by Adam Dossa on 11/06/2013.
//  Copyright (c) 2013 Adam Dossa. All rights reserved.
//
//  Computes Confusion Matrix based on beta* files output by SGD / bound method

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
//#include <libc.h>
#include <netinet/in.h>

#define FEATURE_COLS 360
#define RANK 360
//#define NUM_FILES 3
//#define NUM_HELDOUT_FILES 1
#define NUM_FILES 30700
#define NUM_HELDOUT_FILES 4190
#define NUM_CLASSES 1000
#define EPOCH 12
#define INITIAL_LR 0.001f
#define REG_PARAM 100.0f

float bin2flt(void *src)
{
	typedef union
    {float flt; long lng; }
    flt_t;
	flt_t val;
	val.lng=ntohl( *(int*)src);
	return val.flt;
}

int main(int argc, const char * argv[])
{
    FILE * pFile;
    FILE * outputFile;
    FILE * matrixFile;
    bool debug = false;
    
    //Defaults (for my mac)
    const char * rootDirectory = "/Users/AdamDossa/Documents/Columbia/GRA/Babel/";
    const char * currentDirectory = "/Users/AdamDossa/Documents/XCode/Babel_SGD/Babel_SGD/";
    float regParam = REG_PARAM;
    float initialLR = INITIAL_LR;
    int rank = RANK;
    int epoch = EPOCH;
    
    //Get root directory from args or use default
    if (argc > 1)
    {
        rootDirectory = argv[1];
    }
    if (argc > 2)
    {
        currentDirectory = argv[2];
    }
    if (argc > 3)
    {
        regParam = (float) atof(argv[3]);
    }
    if (argc > 4)
    {
        initialLR = (float) atof(argv[4]);
    }
    if (argc > 5)
    {
        rank = (int) atoi(argv[5]);
    }
    if (argc > 6)
    {
        epoch = (int) atoi(argv[6]);
    }
    
    //We want to find an unused log file name (to some limit)
    char logFileName[200];
    for (int i = 0; i < 100; i++)
    {
        snprintf(logFileName, sizeof(char) * 200,"%s/log_error-%d-%d-%d-%f-%f.txt.%d", currentDirectory, rank, epoch, NUM_HELDOUT_FILES, regParam, initialLR, i);
        outputFile = fopen(logFileName,"r");
        if (outputFile)
        {
            continue;
        } else {
            outputFile = fopen(logFileName,"w");
            break;
        }
    }
    fprintf(outputFile,"Log file: %s\n", logFileName);
    //Also find an unused matrix file name
    for (int i = 0; i < 100; i++)
    {
        snprintf(logFileName, sizeof(char) * 200,"%s/cmatrix-%d-%d-%d-%f-%f.txt.%d", currentDirectory, rank, epoch, NUM_HELDOUT_FILES, regParam, initialLR, i);
        matrixFile = fopen(logFileName,"r");
        if (matrixFile)
        {
            continue;
        } else {
            matrixFile = fopen(logFileName,"w");
            break;
        }
    }
    
    //Now do the same for the heldout data
    //First calculate the number of training examples (since not every file is guaranteed to have 250 rows)
    fprintf(outputFile, "Calculating number of heldout examples\n");
    int noOfHeldoutExamples = 0;
    for (int i = 0; i < NUM_HELDOUT_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/labels/heldout.%d.lab", rootDirectory, i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        n = ntohl(n);
        noOfHeldoutExamples += n;
        fclose(pFile);
    }
    fprintf(outputFile, "No of heldout examples: %d\n", noOfHeldoutExamples);
    
    //First read in the held out labels
    fprintf(outputFile,"Reading in heldout labels\n");
    int readSoFar = 0;
    int * heldLabels = (int *) malloc(sizeof(int) * noOfHeldoutExamples);
    for (int i = 0; i<NUM_HELDOUT_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/labels/heldout.%d.lab", rootDirectory, i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        n = ntohl(n);
        fread(&heldLabels[readSoFar],4,n,pFile);
        for (int j = 0; j < n; j++)
        {
            heldLabels[readSoFar + j] = ntohl(heldLabels[readSoFar + j]);
        }
        readSoFar += n;
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i < noOfHeldoutExamples; i++)
        {
            fprintf(outputFile,"Heldout: Label number: %d has value: %d\n",i,heldLabels[i]);
        }
    }
    fflush(outputFile);
    
    //Now read in the features - hold these in an array of arrays
    fprintf(outputFile,"Reading in heldout features\n");
    readSoFar = 0;
    float ** heldFeatures = (float **) malloc(sizeof(float *) * noOfHeldoutExamples);
    for (int i = 0; i < NUM_HELDOUT_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/features/heldout.%d.fea", rootDirectory, i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        n = ntohl(n);
        for (int j = 0; j < n; j++)
        {
            int m, m_read;
            fread(&m,4,1,pFile);
            m = ntohl(m);
            m_read = m;
            if (m_read > rank)
            {
                m_read = rank;
            }
            heldFeatures[readSoFar + j] = (float *) malloc(sizeof(float) * (m_read + 1));
            fread(heldFeatures[readSoFar + j],sizeof(float),m_read,pFile);
            fseek(pFile, (m - m_read) * sizeof(float), SEEK_CUR);
            for (int k = 0; k < m_read; k++)
            {
                heldFeatures[readSoFar + j][k] = bin2flt(&heldFeatures[readSoFar + j][k]);
            }
            heldFeatures[readSoFar + j][m_read] = 1.0f;
        }
        readSoFar += n;
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i < noOfHeldoutExamples; i++)
        {
            for (int j = 0; j < (rank + 1); j++)
            {
                fprintf(outputFile,"Heldout: Feature coordinate: %d, %d has value: %f\n",i,j,heldFeatures[i][j]);
            }
        }
    }
    fflush(outputFile);
    
    //Now run error calc
    fprintf(outputFile,"Running Error Calculation\n");
    fflush(outputFile);
    
    //Initialize beta parameters, one for each class
    float ** beta = (float **) malloc(sizeof(float *) * NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        beta[i] = (float *) malloc(sizeof(float) * (rank + 1));
        for (int j = 0; j < (rank + 1); j++)
        {
            beta[i][j] = 0.0f;
        }
    }
    
    //Now read in beta files as needed
    char lsName[200];
    FILE *lsFile;
    if (epoch > 0)
    {
        //Initialize using beta file
        snprintf(lsName, sizeof(char) * 200,"%s/beta-%d-%d-%d-%f-%f.out", currentDirectory, rank, epoch - 1, NUM_FILES, regParam, initialLR);
        fprintf(outputFile,"Initializing from file: %s\n", lsName);
        lsFile = fopen(lsName,"rb");
        for (int k = 0; k < NUM_CLASSES; k++)
        {
            fread(beta[k],sizeof(float),rank + 1, lsFile);
        }
        
    }
    fflush(outputFile);
    
    //Now do the error calculation on held out data
    fprintf(outputFile,"Heldout: Calculating Error\n");
    fflush(outputFile);
    
    //Initialize error to 0
    float testError = 0.0f;
    float testAccuracy = 0.0f;
    
    //Store confusion matrix
    int ** confusionMatrix = (int **) malloc(sizeof(int *) * NUM_CLASSES);
    for (int j = 0; j < NUM_CLASSES; j++)
    {
        confusionMatrix[j] = (int *) malloc(sizeof(int) * NUM_CLASSES);
        for (int k = 0; k < NUM_CLASSES; k++)
        {
            confusionMatrix[j][k] = 0;
        }
    }
    
    //Loop over each example
    for (int j = 0; j < noOfHeldoutExamples; j++)
    {
        if (debug) fprintf(outputFile,"Heldout Error - Iterating on example: %d\n",j);
        float dotProductV[NUM_CLASSES];
        float maxDotProduct;
        int maxLabel;
        for (int k = 0; k < NUM_CLASSES; k++)
        {
            float dotProduct = 0.0f;
            for (int d = 0; d < (rank + 1); d++)
            {
                dotProduct += beta[k][d] * heldFeatures[j][d];
            }
            dotProductV[k] = dotProduct;
            if (k==0) {
                maxDotProduct = dotProduct;
                maxLabel = k;
            }
            if (maxDotProduct < dotProduct) {
                maxDotProduct = dotProduct;
                maxLabel = k;
            }
        }
        if (heldLabels[j] != maxLabel) {
            testError += 1.0f;
        } else {
            testAccuracy += 1.0f;
        }
        confusionMatrix[heldLabels[j]][maxLabel] += 1;
    }
    fprintf(outputFile,"Heldout: Error: %f\n", testError/noOfHeldoutExamples);
    fflush(outputFile);
    //Now print the confusion matrix
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            fprintf(matrixFile, "%d, ", confusionMatrix[i][j]);
        }
        fprintf(matrixFile, "\n");
    }
    fflush(matrixFile);
    fclose(outputFile);
    fclose(matrixFile);
    free(heldLabels);
    free(heldFeatures);
    return 0;
}