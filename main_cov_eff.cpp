//
//  main.cpp
//  Babel_SGD
//
//  Created by Adam Dossa on 11/06/2013.
//  Copyright (c) 2013 Adam Dossa. All rights reserved.
//
//  Computes Covariance matrix for input data (faster implementation)
//  NB - some values hardcoded for efficiency

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
//#include <libc.h>
#include <netinet/in.h>

#define FEATURE_COLS 1024
//#define NUM_FILES 100
#define NUM_FILES 30700

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
    bool debug = false;
    bool debugCov = true;
    
    //Defaults (for my mac)
    const char * rootDirectory = "/Users/AdamDossa/Documents/Columbia/GRA/Babel/";
    const char * currentDirectory = "/Users/AdamDossa/Documents/XCode/Babel_SGD/Babel_SGD/log/";
    
    //Get root directory from args or use default
    if (argc > 1)
    {
        rootDirectory = argv[1];
    }
    if (argc > 2)
    {
        currentDirectory = argv[2];
    }
    
    //We want to find an unused log file name (to some limit)
    char logFileName[200];
    for (int i = 0; i < 100; i++)
    {
        snprintf(logFileName, sizeof(char) * 200,"%s/log/log_cov_eff-%d.txt.%d", currentDirectory, NUM_FILES, i);
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
    
    //First calculate the number of training examples (since not every file is guaranteed to have 250 rows)
//    fprintf(outputFile, "Calculating number of training examples\n");
//    int noOfTrainingExamples = 0;
//    for (int i = 0; i < NUM_FILES; i++) {
//        char fName[200];
//        snprintf(fName, sizeof(char) * 200,"%s/labels/train.%d.lab", rootDirectory, i);
//        pFile = fopen(fName,"rb");
//        int n;
//        fread(&n,4,1,pFile);
//        n = ntohl(n);
//        noOfTrainingExamples += n;
//        //printf("File: %d No: %f\n",i,((float) noOfTrainingExamples) / 250.0f);
//        fclose(pFile);
//    }
//    fprintf(outputFile, "No of training examples: %d\n", noOfTrainingExamples);
//    fflush(outputFile);
//    
    
    //Now read in the features - hold these in an array - assume same number as training examples
    fprintf(outputFile,"Reading in features\n");
    int readSoFar = 0;
    float ** features = (float **) malloc(sizeof(float *) * 7675795);
    for (int i = 0; i < 7675795; i++) {
        features[i] = (float *) malloc(sizeof(float) * FEATURE_COLS);
    }
    float featureMeans[FEATURE_COLS];
    for (int i = 0; i< FEATURE_COLS; i++)
    {
        featureMeans[i] = 0.0f;
    }
    for (int i = 0; i < NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/activations/train.%d.fea", rootDirectory, i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        n = ntohl(n);
        for (int j = 0; j < n; j++)
        {
            int m;
            fread(&m,4,1,pFile);
            m = ntohl(m);
            float * tempFeatures = (float *) malloc(sizeof(float) * m);
            fread(&tempFeatures[0],sizeof(float),m,pFile);
            for (int k = 0; k < m; k++)
            {
                features[readSoFar + j][k] = bin2flt(&tempFeatures[k]);
                featureMeans[k] += features[readSoFar + j][k];
            }
            free(tempFeatures);
        }
        readSoFar += n;
        fclose(pFile);
    }
    for (int i = 0; i < FEATURE_COLS; i++)
    {
        //fprintf(outputFile, "Feature Total: %d: %f\n", i, featureMeans[i]);
        featureMeans[i] = featureMeans[i] / 7675795;
    }
    if (debug) {
        for (int i = 0; i < 7675795; i++)
        {
            for (int j = 0; j < FEATURE_COLS; j++)
            {
                fprintf(outputFile,"Train: Feature coordinate: %d, %d has value: %f\n",i,j,features[i][j]);
            }
        }
        for (int i = 0; i < FEATURE_COLS; i++)
        {
            fprintf(outputFile, "Train: Feature coordinate: %d has mean: %f\n",i,featureMeans[i]);
        }
    }
    fflush(outputFile);
    
    //Now center data using featureMeans - not necessary for PCABuildBasis
    for (int i = 0; i < 7675795; i++)
    {
        for (int j = 0; j < FEATURE_COLS; j++)
        {
            features[i][j] = features[i][j] - featureMeans[j];
        }
    }
    if (debug) {
        for (int i = 0; i < 7675795; i++)
        {
            for (int j = 0; j < FEATURE_COLS; j++)
            {
                fprintf(outputFile,"Train: Mean Feature coordinate: %d, %d has centered value: %f\n",i,j,features[i][j]);
            }
        }
    }
    fflush(outputFile);
    
    //Now compute covariance matrix
    float ** covMat = (float **) malloc(sizeof(float *) * FEATURE_COLS);
    for (int i = 0; i < FEATURE_COLS; i++)
    {
        covMat[i] = (float *) malloc(sizeof(float) * FEATURE_COLS);
    }
    
    for (int i = 0; i < 7675795; i++)
    {
        //Fill in linear features
        //        for (int j = 0; j < FEATURE_COLS; j++)
        //        {
        //            tempRow[j] = features[j][i] - featureMeans[j];
        //        }
        //Fill in non-linear features
        for (int j = 0; j < FEATURE_COLS; j++)
        {
            for (int k = j; k < FEATURE_COLS; k++)
            {
                covMat[j][k] += features[i][j] * features[i][k];
            }
        }
    }

    //Fill in other triangle
    for (int j = 0; j < FEATURE_COLS; j++)
    {
        for (int k = 0; k < j; k++)
        {
            covMat[j][k] = covMat[k][j];
        }
    }
    
    //Write out covariance to a file
    char fName[200];
    snprintf(fName, sizeof(char) * 200,"%s/cov/cov-%d.cov", currentDirectory, NUM_FILES);
    pFile = fopen(fName,"wb");
    for (int k = 0; k < FEATURE_COLS; k++)
    {
        for (int j = 0; j < FEATURE_COLS; j++)
        {
            fwrite(&covMat[k][j],sizeof(float),1,pFile);
            if (debugCov)
            {
                fprintf(outputFile, "COV: %d, %d: %f\n", k, j, covMat[k][j]);
            }
        }
        fflush(outputFile);
    }
    fflush(outputFile);
    fclose(pFile);
    fclose(outputFile);
    return 0;
}