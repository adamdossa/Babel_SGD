//
//  main.cpp
//  Babel_SGD
//
//  Created by Adam Dossa on 11/06/2013.
//  Copyright (c) 2013 Adam Dossa. All rights reserved.
//
//  Writes out data in format for LSH

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
//#include <libc.h>
#include <netinet/in.h>

#define FEATURE_COLS 360
#define RANK 360
#define NUM_FILES 2
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
    FILE * oFile;
    FILE * outputFile;
    
    //Defaults (for my mac)
    const char * rootDirectory = "/Users/AdamDossa/Documents/Columbia/GRA/Babel/";
    const char * currentDirectory = "/Users/AdamDossa/Documents/XCode/Babel_SGD/Babel_SGD/log/";
    const char * trainFileName = "heldout";
    
    bool doPCA = true;
    bool debugLoadings = true;
    int rank = RANK;
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
        snprintf(logFileName, sizeof(char) * 200,"%s/log/log_reformat-%d.txt.%d", currentDirectory, NUM_FILES, i);
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
    
    double ** loadings;
    
    if (doPCA) {
        //First read in the loadings, must have been generated for correct number of input files (dubious)
        fprintf(outputFile, "Reading in loadings\n");
        loadings = (double **) malloc(sizeof(double *) * FEATURE_COLS);
        char loadName[200];
        snprintf(loadName, sizeof(char) * 200,"%s/loadings/loadings-%d.load", currentDirectory, NUM_FILES);
        pFile = fopen(loadName,"rb");
        for (int i = 0; i < FEATURE_COLS; i++)
        {
            loadings[i] = (double *) malloc(sizeof(double) * FEATURE_COLS);
            fread(loadings[i], sizeof(double), FEATURE_COLS, pFile);
        }
        if (debugLoadings)
        {
            for (int i = 0; i < FEATURE_COLS; i++)
            {
                for (int j = 0; j < FEATURE_COLS; j++)
                {
                    fprintf(outputFile, "Loadings for %d, %d: %f\n", i, j, loadings[i][j]);
                }
            }
        }
        fclose(pFile);
        fflush(outputFile);
    }
    
    //First calculate the number of training examples (since not every file is guaranteed to have 250 rows)
    fprintf(outputFile, "Calculating number of training examples\n");
    int noOfTrainingExamples = 0;
    for (int i = 0; i < NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/labels/%s.%d.lab", rootDirectory, trainFileName, i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        n = ntohl(n);
        noOfTrainingExamples += n;
        //printf("File: %d No: %f\n",i,((float) noOfTrainingExamples) / 250.0f);
        fclose(pFile);
    }
    fprintf(outputFile, "No of examples: %d\n", noOfTrainingExamples);
    fflush(outputFile);

    //Write out reformatted data to a file
    char fName[200];
    if (doPCA) {
        snprintf(fName, sizeof(char) * 200,"%s/lshFeatures/features_pca-%s-%d.lsh", currentDirectory, trainFileName, NUM_FILES);
    } else {
        snprintf(fName, sizeof(char) * 200,"%s/lshFeatures/features-%s-%d.lsh", currentDirectory, trainFileName, NUM_FILES);
    }
    oFile = fopen(fName,"wb");
    unsigned int size = 4;
    unsigned int rows = noOfTrainingExamples;
    unsigned int cols;
    if (doPCA) {
        cols = rank;
    } else {
        cols = FEATURE_COLS;
    }
    fwrite(&size, sizeof(unsigned),1,oFile);
    fwrite(&rows, sizeof(unsigned),1,oFile);
    fwrite(&cols, sizeof(unsigned),1,oFile);

    //Now read in the features - hold these in an array of arrays - assume same number as training examples
    //We apply loadings as we store each sample
    fprintf(outputFile,"Reading in features\n");
//    readSoFar = 0;
//    float ** features = (float **) malloc(sizeof(float *) * noOfTrainingExamples);
    for (int i = 0; i < NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/features/%s.%d.fea", rootDirectory, trainFileName, i);
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
            //features[readSoFar + j] = (float *) malloc(sizeof(float) * (rank + 1));
            //fread(features[readSoFar + j],sizeof(float),m,pFile);
            if (doPCA) {
                for (int k = 0; k < rank; k++)
                {
                    float featureI = 0.0f;
                    for (int p = 0; p < m; p++)
                    {
                        featureI += ((float) loadings[p][k]) * bin2flt(&tempFeatures[p]);
                    }
                    fwrite(&featureI, sizeof(float), 1, oFile);
                    //features[readSoFar + j][k] = featureI;
                }
            } else {
                for (int k = 0; k < m; k++)
                {
                    tempFeatures[k] = bin2flt(&tempFeatures[k]);
                }
                fwrite(tempFeatures, sizeof(float), m, oFile);
            }
            free(tempFeatures);
            //features[readSoFar + j][rank] = 1.0f;
        }
        //readSoFar += n;
        fclose(pFile);
    }
    fflush(outputFile);
    free(loadings);
    fclose(oFile);
    fclose(outputFile);
    return 0;
}