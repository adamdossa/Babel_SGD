//
//  main.cpp
//  Babel_SGD
//
//  Created by Adam Dossa on 11/06/2013.
//  Copyright (c) 2013 Adam Dossa. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <libc.h>
#include <netinet/in.h>

#define FEATURE_COLS 360
//#define NUM_FILES 1
//#define NUM_HELDOUT_FILES 1
#define NUM_FILES 30700
#define NUM_HELDOUT_FILES 4190
#define NUM_CLASSES 1000

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
    
    //Defaults (for my mac)
    const char * rootDirectory = "/Users/AdamDossa/Documents/Columbia/GRA/Babel/";
    const char * currentDirectory = "/Users/AdamDossa/Documents/XCode/Babel_SGD/Babel_SGD/";
    
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
        snprintf(logFileName, sizeof(char) * 200,"%s/log_prior-%d.txt.%d", currentDirectory, NUM_FILES, i);
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
    
    //First read in the labels
    fprintf(outputFile, "Reading in labels\n");
    int readSoFar = 0;
    int * labels = (int *) malloc(sizeof(int) * noOfTrainingExamples);
    for (int i = 0; i < NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/labels/train.%d.lab", rootDirectory, i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        n = ntohl(n);
        fread(&labels[readSoFar],4,n,pFile);
        for (int j = 0; j < n; j++)
        {
            labels[readSoFar + j] = ntohl(labels[readSoFar + j]);
        }
        readSoFar += n;
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i < noOfTrainingExamples; i++)
        {
            fprintf(outputFile,"Train: Label number: %d has value: %d\n",i,labels[i]);
        }
    }
    fflush(outputFile);
    
    //Now read in the features - hold these in an array of arrays - assume same number as training examples
    fprintf(outputFile,"Reading in features\n");
    readSoFar = 0;
    float ** features = (float **) malloc(sizeof(float *) * noOfTrainingExamples);
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
            features[readSoFar + j] = (float *) malloc(sizeof(float) * (m + 1));
            fread(features[readSoFar + j],sizeof(float),m,pFile);
            for (int k = 0; k < m; k++)
            {
                features[readSoFar + j][k] = bin2flt(&features[readSoFar + j][k]);
            }
            features[readSoFar + j][360] = 1.0f;
        }
        readSoFar += n;
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i < noOfTrainingExamples; i++)
        {
            for (int j = 0; j<(FEATURE_COLS + 1); j++)
            {
                fprintf(outputFile,"Train: Feature coordinate: %d, %d has value: %f\n",i,j,features[i][j]);
            }
        }
    }
    fflush(outputFile);
    
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
    readSoFar = 0;
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
            int m;
            fread(&m,4,1,pFile);
            m = ntohl(m);
            heldFeatures[readSoFar + j] = (float *) malloc(sizeof(float) * (m + 1));
            fread(heldFeatures[readSoFar + j],sizeof(float),m,pFile);
            for (int k = 0; k < m; k++)
            {
                heldFeatures[readSoFar + j][k] = bin2flt(&heldFeatures[readSoFar + j][k]);
            }
            heldFeatures[readSoFar + j][360] = 1.0f;
        }
        readSoFar += n;
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i < noOfHeldoutExamples; i++)
        {
            for (int j = 0; j < (FEATURE_COLS + 1); j++)
            {
                fprintf(outputFile,"Heldout: Feature coordinate: %d, %d has value: %f\n",i,j,heldFeatures[i][j]);
            }
        }
    }
    fflush(outputFile);
    
    //Now run Prior calcs
    fprintf(outputFile,"Running Baseline Prior Calculation\n");
    fflush(outputFile);
    
    float * priors = (float *) malloc(sizeof(float) * NUM_CLASSES);
    
    for (int i = 0; i < noOfTrainingExamples; i++) {
        priors[labels[i]] += 1;
    }
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        priors[i] = priors[i] / noOfTrainingExamples;
    }
    
    
    //Now do the log-likelihood calculation (on training data)
    fprintf(outputFile,"Train: Calculating Likelihood\n");
    fflush(outputFile);
    
    float logLikelihood = 0.0f;
    //Loop over each example
    for (int j = 0; j < noOfTrainingExamples; j++)
    {
        if (debug) fprintf(outputFile,"Train Likelihood: Iterating on example: %d\n",j);
        float priorProb = priors[labels[j]];
        logLikelihood += logf(priorProb);
    }
    fprintf(outputFile,"Train: LogLikelihood: %f\n", logLikelihood);
    
    //Now do the log-likelihood calculation on held out data
    fprintf(outputFile,"Heldout: Calculating Likelihood\n");
    fflush(outputFile);

    //Initialize logLikelihood to 0
    float heldLogLikelihood = 0.0f;
    
    //Loop over each example
    for (int j = 0; j < noOfHeldoutExamples; j++)
    {
        if (debug) fprintf(outputFile,"Heldout Likelihood - Iterating on example: %d\n",j);
        float priorProb = priors[heldLabels[j]];
        heldLogLikelihood += logf(priorProb);
    }
    fprintf(outputFile,"Heldout: LogLikelihood: %f\n", heldLogLikelihood);
    

    fclose(pFile);
    fflush(outputFile);
    
    free(labels);
    free(features);
    free(heldLabels);
    free(heldFeatures);
    return 0;
}