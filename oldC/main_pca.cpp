//
//  main.cpp
//  Babel_SGD
//
//  Created by Adam Dossa on 11/06/2013.
//  Copyright (c) 2013 Adam Dossa. All rights reserved.
//
//  Computes PCA loadings for input data - not used, done in Matlab instead

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
//#include <libc.h>
#include <netinet/in.h>
#include "src/alglibmisc.h"
#include "src/dataanalysis.h"

#define FEATURE_COLS 360
//#define NUM_FILES 1
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
    bool debugEig = true;
    alglib::real_2d_array ptInput;
    
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
        snprintf(logFileName, sizeof(char) * 200,"%s/log_pca-%d.txt.%d", currentDirectory, NUM_FILES, i);
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
    fprintf(outputFile, "Calculating number of training examples\n");
    int noOfTrainingExamples = 0;
    for (int i = 0; i < NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/labels/train.%d.lab", rootDirectory, i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        n = ntohl(n);
        noOfTrainingExamples += n;
        //printf("File: %d No: %f\n",i,((float) noOfTrainingExamples) / 250.0f);
        fclose(pFile);
    }
    fprintf(outputFile, "No of training examples: %d\n", noOfTrainingExamples);
    fflush(outputFile);
    
    //Set input size
    ptInput.setlength(noOfTrainingExamples, FEATURE_COLS);
    
    //Now read in the features - hold these in an array - assume same number as training examples
    fprintf(outputFile,"Reading in features\n");
    int readSoFar = 0;
    //double * features = (double *) malloc(sizeof(double) * noOfTrainingExamples * FEATURE_COLS);
    //double featureMeans[FEATURE_COLS];
//    for (int i = 0; i< FEATURE_COLS; i++)
//    {
//        featureMeans[i] = 0.0;
//    }
    for (int i = 0; i < NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/features/train.%d.fea", rootDirectory, i);
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
                ptInput[readSoFar + j][k] = static_cast<double>(bin2flt(&tempFeatures[k]));
//                featureMeans[k] += features[readSoFar + k];
            }
            free(tempFeatures);
        }
        readSoFar += n;
        fclose(pFile);
    }
//    for (int i = 0; i < FEATURE_COLS; i++)
//    {
//        featureMeans[i] = featureMeans[i] / noOfTrainingExamples;
//    }
//    if (debug) {
//        for (int i = 0; i < (noOfTrainingExamples * FEATURE_COLS); i++)
//        {
//            fprintf(outputFile,"Train: Feature coordinate: %d has value: %f\n",i,features[i]);
//        }
//        for (int i = 0; i < FEATURE_COLS; i++)
//        {
//            fprintf(outputFile, "Train: Feature coordinate: %d has mean: %f\n",i,featureMeans[i]);
//        }
//    }
    fflush(outputFile);
    
    //Now center data using featureMeans - not necessary for PCABuildBasis
//    for (int i = 0; i < noOfTrainingExamples; i++)
//    {
//        for (int j = 0; j < FEATURE_COLS; j++)
//        {
//            features[(i * FEATURE_COLS) + j] = features[(i * FEATURE_COLS) + j] - featureMeans[j];
//        }
//    }
//    if (debug) {
//        for (int i = 0; i < (noOfTrainingExamples * FEATURE_COLS); i++)
//        {
//            fprintf(outputFile,"Train: Feature coordinate: %d has centered value: %f\n",i,features[i]);
//        }
//    }
//    fflush(outputFile);
    
    //Now run PCA
    fprintf(outputFile,"Running PCA\n");
    fflush(outputFile);
    
    //Run PCA on input
//    alglib::real_2d_array ptInput;
//    ptInput.setcontent(noOfTrainingExamples, FEATURE_COLS , features);
//    free(features);
    fprintf(outputFile, "ptInput created\n");
    fflush(outputFile);
    
    alglib::ae_int_t info;
    alglib::real_1d_array eigValues;
    alglib::real_2d_array eigVectors;
    if (debug) {
        for (int i = 0; i < noOfTrainingExamples; i++)
        {
            for (int j = 0; j < FEATURE_COLS; j++)
            {
                fprintf(outputFile, "ptInput: %d, %d, %f\n", i,j, ptInput[i][j]);
            }
        }
    }
    try {
        pcabuildbasis(ptInput, noOfTrainingExamples, FEATURE_COLS, info, eigValues, eigVectors);
    } catch (alglib::ap_error& err) {
        std::cout << err.msg;
    }
    //pcabuildbasis(ptInput, noOfTrainingExamples, FEATURE_COLS, info, eigValues, eigVectors);
    if (debugEig) {
        for (int i = 0; i < FEATURE_COLS; i++)
        {
            fprintf(outputFile, "EIGVALS: %f\n", eigValues[i]);
        }
    }
    fflush(outputFile);
    
    //Now compute scores
//    alglib::real_2d_array scores;
//    scores.setlength(noOfTrainingExamples, FEATURE_COLS);
//    alglib::rmatrixgemm(noOfTrainingExamples, FEATURE_COLS ,FEATURE_COLS, 1, ptInput, 0,0,0, eigVectors, 0,0,0, 0, scores,0,0);
//    if (debug)
//    {
//        for (int i = 0; i < noOfTrainingExamples; i++)
//        {
//            for (int j = 0; j < FEATURE_COLS; j++)
//            {
//                fprintf(outputFile, "PCA SCORES: %d, %d, %f\n", i,j,scores[i][j]);
//            }
//        }
//    }
//    fclose(outputFile);
    
    //Write out loadings to a file
    char fName[200];
    snprintf(fName, sizeof(char) * 200,"%s/loadings-%d.load", currentDirectory, NUM_FILES);
    pFile = fopen(fName,"wb");
    for (int k = 0; k < FEATURE_COLS; k++)
    {
        for (int j = 0; j < FEATURE_COLS; j++)
        {
            fwrite(&eigVectors[k][j],sizeof(double),1,pFile);
        }
    }
    fclose(pFile);
    
    fclose(outputFile);
    return 0;
}