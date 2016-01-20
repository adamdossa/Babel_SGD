//
//  main.cpp
//  Babel_SGD
//
//  Created by Adam Dossa on 11/06/2013.
//  Copyright (c) 2013 Adam Dossa. All rights reserved.
//
//  Computes Covariance matrix for input data concatenated with non-linear features (more efficient version)

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
//#include <libc.h>
#include <netinet/in.h>

#define FEATURE_COLS 60
#define NUM_FILES 1
//#define NUM_FILES 30700

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
        snprintf(logFileName, sizeof(char) * 200,"%s/log_covnl-%d.txt.%d", currentDirectory, NUM_FILES, i);
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
    int noOfTrainingExamples = FEATURE_COLS;
    fprintf(outputFile, "No of training examples: %d\n", noOfTrainingExamples);
    fflush(outputFile);
    
    //Now read in the features - hold these in an array - assume same number as training examples
    fprintf(outputFile,"Reading in features\n");
    float dimT = (((FEATURE_COLS + 1.0f)/2.0f) * FEATURE_COLS);
    int dim = (int) dimT;
    fprintf(outputFile, "Dimension: %d\n",dim);
    fflush(outputFile);
    
    float ** features = (float **) malloc(sizeof(float *) * noOfTrainingExamples);
    float featureMeans[dim];
    for (int i = 0; i< dim; i++)
    {
        featureMeans[i] = 0.0f;
    }
    int m;
    for (int i = 0; i < NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/pcaFeatures/feaPCAT.mat", rootDirectory);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        //n = ntohl(n);
        for (int j = 0; j < n; j++)
        {
            fread(&m,4,1,pFile);
            //m = ntohl(m);
            features[j] = (float *) malloc(sizeof(float) * m);
            //float * tempFeatures = (float *) malloc(sizeof(float) * m);
            fread(features[j],sizeof(float),m,pFile);
//            for (int k = 0; k < m; k++)
//            {
//                //features[readSoFar + j][k] = bin2flt(&tempFeatures[k]);
//                featureMeans[j] += features[j][k];
//            }
            
            //free(tempFeatures);
        }
        //readSoFar += n;
        fclose(pFile);
    }
    
    //Now calculate remaining means
    int ind = 0;
    for (int i = 0; i < FEATURE_COLS; i++)
    {
        for (int j = i; j < FEATURE_COLS; j++)
        {
            float tMean = 0.0f;
            for (int k = 0; k < m; k++)
            {
                tMean += features[j][k] * features[i][k];
            }
            featureMeans[ind] = tMean;
            ind += 1;
        }
    }
    
    for (int i = 0; i < dim; i++)
    {
        featureMeans[i] = featureMeans[i] / m;
        fprintf(outputFile, "Mean: %d: %f\n", i, featureMeans[i]);
    }
    fflush(outputFile);
    
    //Initialize covariance matrix
    float ** covMat = (float **) malloc(sizeof(float *) * dim);
    for (int i = 0; i < dim; i++)
    {
        covMat[i] = (float *) malloc(sizeof(float) * dim);
        for (int j = 0; j < dim; j++)
        {
            covMat[i][j] = 0.0f;
        }
    }
    
    //Calculate covar matrix
    float * tempRow = (float *) malloc(sizeof(float) * m);
    for (int i = 0; i < m; i++)
    {
        //Fill in linear features
//        for (int j = 0; j < FEATURE_COLS; j++)
//        {
//            tempRow[j] = features[j][i] - featureMeans[j];
//        }
        //Fill in non-linear features
        int quadInd = 0;
        for (int j = 0; j < FEATURE_COLS; j++)
        {
            for (int k = j; k < FEATURE_COLS; k++)
            {
                tempRow[quadInd] = (features[j][i] * features[k][i]) - featureMeans[quadInd];
                quadInd += 1;
            }
        }
        for (int j = 0; j < dim; j++)
        {
            for (int k = j; k < dim; k++)
            {
                covMat[j][k] += tempRow[j] * tempRow[k];
            }
        }
    }
    free(tempRow);
    
    //Fill in other triangle
    for (int j = 0; j < dim; j++)
    {
        for (int k = 0; k < j; k++)
        {
            covMat[j][k] = covMat[k][j];
        }
    }
    
    //Write out covariance to a file
    char fName[200];
    snprintf(fName, sizeof(char) * 200,"%s/covnlsmall-%d.cov", currentDirectory, dim);
    pFile = fopen(fName,"wb");
    for (int k = 0; k < dim; k++)
    {
        for (int j = 0; j < dim; j++)
        {
            fwrite(&covMat[k][j],sizeof(float),1,pFile);
            if (debugCov)
            {
                fprintf(outputFile, "COV: %d, %d: %f\n", k, j, covMat[k][j]);
            }
        }
        fflush(outputFile);
    }
    free(covMat); // Should call for each row really
    free(features); // Should call for each row really
    fflush(outputFile);
    fclose(pFile);
    fclose(outputFile);
    return 0;
}