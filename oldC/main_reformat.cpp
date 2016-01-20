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
        snprintf(logFileName, sizeof(char) * 200,"%s/log_reformat-%d.txt.%d", currentDirectory, NUM_FILES, i);
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

    //Write out reformatted data to a file
    char fName[200];
    snprintf(fName, sizeof(char) * 200,"%s/features-%d.cov", currentDirectory, NUM_FILES);
    oFile = fopen(fName,"wb");
    unsigned int size = 4;
    unsigned int rows = noOfTrainingExamples;
    unsigned int cols = FEATURE_COLS;
    fwrite(&size, sizeof(unsigned),1,oFile);
    fwrite(&rows, sizeof(unsigned),1,oFile);
    fwrite(&cols, sizeof(unsigned),1,oFile);

//    for (int k = 0; k < FEATURE_COLS; k++)
//    {
//        for (int j = 0; j < FEATURE_COLS; j++)
//        {
//            fwrite(&covMat[k][j],sizeof(float),1,pFile);
//            if (debugCov)
//            {
//                fprintf(outputFile, "COV: %d, %d: %f\n", k, j, covMat[k][j]);
//            }
//        }
//        fflush(outputFile);
//    }

    
    //Now read in the features - hold these in an array - assume same number as training examples
    fprintf(outputFile,"Reading in features\n");
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
                tempFeatures[k] = bin2flt(&tempFeatures[k]);
            }
            fwrite(tempFeatures, sizeof(float), m, oFile);
            free(tempFeatures);
        }
        fclose(pFile);
    }
    fclose(oFile);
    fclose(outputFile);
    return 0;
}