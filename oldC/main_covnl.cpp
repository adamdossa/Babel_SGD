//
//  main.cpp
//  Babel_SGD
//
//  Created by Adam Dossa on 11/06/2013.
//  Copyright (c) 2013 Adam Dossa. All rights reserved.
//
//  Computes Covariance matrix for input data concatenated with non-linear features

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
    float dimT = FEATURE_COLS + (((FEATURE_COLS + 1.0f)/2.0f) * FEATURE_COLS);
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
            for (int k = 0; k < m; k++)
            {
                //features[readSoFar + j][k] = bin2flt(&tempFeatures[k]);
                featureMeans[j] += features[j][k];
            }
            
            //free(tempFeatures);
        }
        //readSoFar += n;
        fclose(pFile);
    }
    
    //Now calculate remaining means
    int ind = FEATURE_COLS;
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
    
    for (int i = 0; i< dim; i++)
    {
        featureMeans[i] = featureMeans[i] / m;
        fprintf(outputFile, "Mean: %d: %f\n", i, featureMeans[i]);
    }
    
    //Initialize covariance matrix
    float ** covMat = (float **) malloc(sizeof(float *) * dim);
    for (int i = 0; i < dim; i++)
    {
        covMat[i] = (float *) malloc(sizeof(float) * dim);
    }
    
    //Ignore initial block of cov matrix for now
    int x_co = FEATURE_COLS;
    int y_co = FEATURE_COLS;
    for (int i = 0; i < FEATURE_COLS; i++)
    {
        for (int j = i; j < FEATURE_COLS; j++)
        {
            fprintf(outputFile, "Doing covar for %d, %d\n", x_co, y_co);
            fflush(outputFile);
            y_co = FEATURE_COLS;// + (x_co - FEATURE_COLS);
            //Get relevant row for first coordinate
            float * tempRow = (float *) malloc(sizeof(float) * m);
            for (int k = 0; k < m; k++)
            {
                tempRow[k] = (features[j][k] * features[i][k]) - featureMeans[x_co];
            }
            for (int i1 = 0; i1 < FEATURE_COLS; i1++)
            {
                for (int j1 = i1; j1 < FEATURE_COLS; j1++)
                {
                    if (y_co >= x_co) {
                        float cVal = 0.0f;
                        for (int k1 = 0; k1 < m; k1++)
                        {
                            float featY = (features[j1][k1] * features[i1][k1]) - featureMeans[y_co];
                            cVal += tempRow[k1] * featY;
                        }
                        covMat[x_co][y_co] = cVal;
                        covMat[y_co][x_co] = cVal;
                    }
                    y_co += 1;
                }
            }
            x_co += 1;
        }
    }
    
    //Now do initial block
    for (int i = 0; i < FEATURE_COLS; i++)
    {
        fprintf(outputFile, "Doing covar2 for %d\n", i);
        fflush(outputFile);
        for (int j = i; j < FEATURE_COLS; j++)
        {
            float cVal = 0.0f;
            for (int k = 0; k < m; k++)
            {
                cVal += (features[i][k] - featureMeans[i]) * (features[j][k] - featureMeans[j]);
            }
            covMat[i][j] = cVal;
            covMat[j][i] = cVal;
        }
    }
    
    //Now do top-right block
    for (int i = 0; i < FEATURE_COLS; i++)
    {
        y_co = FEATURE_COLS;
        fprintf(outputFile, "Doing covar3 for %d, %d\n", i, y_co);
        fflush(outputFile);
        for (int y0 = 0; y0 < FEATURE_COLS; y0++)
        {
            for (int y1 = y0; y1 < FEATURE_COLS; y1++)
            {
                float cVal = 0.0f;
                for (int k = 0; k < m; k++)
                {
                    float featY = (features[y1][k] * features[y0][k]) - featureMeans[y_co];
                    cVal += (features[i][k] - featureMeans[i]) * featY;
                    
                    //cVal += (features[i][k] - featureMeans[i]) * (features[j][k] - featureMeans[j]);
                }
                covMat[i][y_co] = cVal;
                covMat[y_co][i] = cVal;
                y_co += 1;
            }
        }
    }
    
    
    //Write out covariance to a file
    char fName[200];
    snprintf(fName, sizeof(char) * 200,"%s/covnl-%d.cov", currentDirectory, dim);
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
    fflush(outputFile);
    fclose(pFile);
    fclose(outputFile);
    return 0;
}