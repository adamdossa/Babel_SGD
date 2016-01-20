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
#define NUM_FILES 1
//#define NUM_HELDOUT_FILES 100
//#define NUM_FILES 30700
#define NUM_HELDOUT_FILES 4190
#define NUM_CLASSES 1000
#define NUM_EPOCHS 20
#define INITIAL_LR 0.1f
#define REG_PARAM 1.0f

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
    bool debugReg = false;
    bool timeLoops = false;
    
    //Defaults (for my mac)
    const char * rootDirectory = "/Users/AdamDossa/Documents/Columbia/GRA/Babel/";
    const char * currentDirectory = "/Users/AdamDossa/Documents/XCode/Babel_SGD/Babel_SGD/";
    float regParam = REG_PARAM;
    float initialLR = INITIAL_LR;
    
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
    
    //We want to find an unused log file name (to some limit)
    char logFileName[200];
    for (int i = 0; i < 100; i++)
    {
        snprintf(logFileName, sizeof(char) * 200,"%s/log-%d-%d-%f-%f.txt.%d", currentDirectory, NUM_EPOCHS, NUM_FILES, regParam, initialLR, i);
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
    
    //Now run SGD
    fprintf(outputFile,"Running SGD\n");
    fflush(outputFile);
    
    //Initialize beta parameters, one for each class
    float ** beta = (float **) malloc(sizeof(float *) * NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        beta[i] = (float *) malloc(sizeof(float) * (FEATURE_COLS + 1));
        for (int j = 0; j < (FEATURE_COLS + 1); j++)
        {
            beta[i][j] = 0.0f;
        }
    }
    
    //Create timing variables, used if timeLoops = true
    time_t t0, t1;
    clock_t c0, c1;
    if (timeLoops) {
        t0 = time(NULL);
        c0 = clock();
    }
    
    //Add new resume functionality
    //Check for any beta* files of appropriate name, and initialize using latest epoch
    int startingEpoch = 0;
    char lsName[200];
    snprintf(lsName, sizeof(char) * 200,"ls %s/beta-%s-%d-%f-%f.out | grep 'beta' | cut -d'-' -f2", currentDirectory, "*", NUM_FILES, regParam, initialLR);
    FILE *lsFile = popen(lsName, "r" );
    sleep(1);
    if (lsFile == 0 ) {
        fprintf(outputFile, "Could not execute ls command\n" );
        return 1;
    }
    const int BUFSIZE = 1000;
    char lsBuffer[BUFSIZE];
    while(fgets(lsBuffer, BUFSIZE,  lsFile))
    {
        fprintf(outputFile, "Found file for resume: %s", lsBuffer);
        startingEpoch = 1 + atoi(lsBuffer);
    }
    pclose(lsFile);
    
    //Now read in beta files as needed
    if (startingEpoch > 0)
    {
        //Initialize using beta file
        snprintf(lsName, sizeof(char) * 200,"%s/beta-%d-%d-%f-%f.out", currentDirectory, startingEpoch - 1, NUM_FILES, regParam, initialLR);
        fprintf(outputFile,"Initializing from file: %s\n", lsName);
        lsFile = fopen(lsName,"rb");
        for (int k = 0; k < NUM_CLASSES; k++)
        {
            fread(beta[k],sizeof(float),FEATURE_COLS + 1, lsFile);
        }

    }
    fflush(outputFile);
    
    //Start looping for NUM_EPOCHS
    for (int e = startingEpoch; e < NUM_EPOCHS; e++)
    {
        fprintf(outputFile,"SGD Epoch: %d\n", e);
        fflush(outputFile);
        //Calculate learning rate for epoch
        //Loop over examples
        float epochExamples = (e * noOfTrainingExamples);
        for (int j = 0; j < noOfTrainingExamples; j++)
        {
            float learningRate = initialLR/(1 + ((epochExamples + j)/(float) noOfTrainingExamples));
            if (debug)
            {
                fprintf(outputFile,"Train: Iterating on example: %d\n",j);
                fprintf(outputFile,"Learning Rate: %f\n", learningRate);
            }
            
            //Generate loop times (every 1000 loops)
            if (timeLoops) {
                float batchCounter = ((float) j) / 1000.0f;
                double integral;
                double fractional = modf(batchCounter, &integral);
                if (fractional == 0.0f) {
                    fprintf(outputFile,"Train: Loop: %d\n",j);
                    t1 = time(NULL);
                    c1 = clock();
                    fprintf(outputFile,"Train: Elapsed wall clock time: %ld\n", (long) (t1 - t0));
                    fprintf (outputFile, "Train: Elapsed CPU time: %f\n", (float) (c1 - c0)/CLOCKS_PER_SEC);
                    t0 = time(NULL);
                    c0 = clock();
                    fflush(outputFile);
                }
            }
            //Initialize z (the denominator of our probability function:
            //sum_i(exp(beta_i.x_i))
            float z = 0.0f;
            //We store each dot product of beta_i.x_i as we go
            float dotProductV[NUM_CLASSES];
            //In order to avoid overflows, we use the identity:
            //log(sum_i(exp(beta_i.x_i))) = m + log(sum_i(exp(beta_i.x_i - m)))
            //where m is the maxDotProduct
            float maxDotProduct;
            for (int k = 0; k < NUM_CLASSES; k++)
            {
                float dotProduct = 0.0f;
                for (int d = 0; d < (FEATURE_COLS + 1); d++)
                {
                    dotProduct += beta[k][d] * features[j][d];
                }
                dotProductV[k] = dotProduct;
                if (k==0) maxDotProduct = dotProduct;
                if (maxDotProduct < dotProduct) {
                    maxDotProduct = dotProduct;
                }
            }
            //Now calculate z as above
            for (int k = 0; k < NUM_CLASSES; k++)
            {
                z += expf(dotProductV[k] - maxDotProduct);
            }
            z = maxDotProduct + logf(z);
            //Update each beta
            for (int c = 0; c < NUM_CLASSES; c++)
            {
                //Calculate p(c_c|x_j, beta_c)
                float p_c_x_c = expf(dotProductV[c] - z);

                //Update rule depends on indicator function (Gaussian Prior)
                if (c == labels[j])
                {
                    for (int d = 0; d < (FEATURE_COLS + 1); d++)
                    {
                        beta[c][d] += learningRate*(features[j][d]*(1.0f - p_c_x_c) - (regParam * beta[c][d])/noOfTrainingExamples);
                    }
                } else {
                    for (int d = 0; d < (FEATURE_COLS + 1); d++)
                    {
                        beta[c][d] += learningRate*(features[j][d]*(0.0f - p_c_x_c) - (regParam * beta[c][d])/noOfTrainingExamples);
                    }                    
                }
            }
        }
        
        //Now do the log-likelihood calculation (on training data)
        fprintf(outputFile,"Train: Calculating Likelihood\n");
        fflush(outputFile);

        //Initialize logLikelihood to 0
        float logLikelihood = 0.0f;
        if (timeLoops) {
            t0 = time(NULL);
            c0 = clock();
        }
        
        //Loop over each example
        for (int j = 0; j < noOfTrainingExamples; j++)
        {
            if (debug) fprintf(outputFile,"Train Likelihood: Iterating on example: %d\n",j);
            if (timeLoops) {
                float batchCounter = ((float) j) / 1000.0f;
                double integral;
                double fractional = modf(batchCounter, &integral);
                if (fractional == 0.0f) {
                    fprintf(outputFile,"Train Likelihood Loop: %d\n",j);
                    t1 = time(NULL);
                    c1 = clock();
                    fprintf(outputFile,"Train Likelihood: Elapsed wall clock time: %ld\n", (long) (t1 - t0));
                    fprintf(outputFile, "Train Likelihood: Elapsed CPU time: %f\n", (float) (c1 - c0)/CLOCKS_PER_SEC);
                    t0 = time(NULL);
                    c0 = clock();
                    fflush(outputFile);
                }
            }
            float z = 0.0f;
            float dotProductV[NUM_CLASSES];
            float maxDotProduct;
            for (int k = 0; k < NUM_CLASSES; k++)
            {
                float dotProduct = 0.0f;
                for (int d = 0; d < (FEATURE_COLS + 1); d++)
                {
                    dotProduct += beta[k][d] * features[j][d];
                }
                dotProductV[k] = dotProduct;
                if (k==0) maxDotProduct = dotProduct;
                if (maxDotProduct < dotProduct) {
                    maxDotProduct = dotProduct;
                }
            }
            for (int k = 0; k < NUM_CLASSES; k++)
            {
                z += expf(dotProductV[k] - maxDotProduct);
            }
            float regTerm = 0.0f;
            for (int d = 0; d < (FEATURE_COLS + 1); d++)
            {
                regTerm += powf(beta[labels[j]][d],2);
            }
            logLikelihood += dotProductV[labels[j]] - (maxDotProduct + logf(z)) - ((0.5f * regParam * regTerm)/(float) noOfTrainingExamples);
        }
        fprintf(outputFile,"Train: LogLikelihood: %f\n", logLikelihood);
        
        //Now do the log-likelihood calculation on held out data
        fprintf(outputFile,"Heldout: Calculating Likelihood\n");
        fflush(outputFile);
        //Initialize logLikelihood to 0
        float heldLogLikelihood = 0.0f;
        if (timeLoops) {
            t0 = time(NULL);
            c0 = clock();
        }
        
        //Loop over each example
        for (int j = 0; j < noOfHeldoutExamples; j++)
        {
            if (debug) fprintf(outputFile,"Heldout Likelihood - Iterating on example: %d\n",j);
            if (timeLoops) {
                float batchCounter = ((float) j) / 1000.0f;
                double integral;
                double fractional = modf(batchCounter, &integral);
                if (fractional == 0.0f) {
                    fprintf(outputFile,"Heldout Likelihood Loop: %d\n",j);
                    t1 = time(NULL);
                    c1 = clock();
                    fprintf(outputFile,"Heldout Likelihood: Elapsed wall clock time: %ld\n", (long) (t1 - t0));
                    fprintf(outputFile, "Heldout Likelihood: Elapsed CPU time: %f\n", (float) (c1 - c0)/CLOCKS_PER_SEC);
                    t0 = time(NULL);
                    c0 = clock();
                    fflush(outputFile);
                }
            }
            float z = 0.0f;
            float dotProductV[NUM_CLASSES];
            float maxDotProduct;
            for (int k = 0; k < NUM_CLASSES; k++)
            {
                float dotProduct = 0.0f;
                for (int d = 0; d < (FEATURE_COLS + 1); d++)
                {
                    dotProduct += beta[k][d] * heldFeatures[j][d];
                }
                dotProductV[k] = dotProduct;
                if (k==0) maxDotProduct = dotProduct;
                if (maxDotProduct < dotProduct) {
                    maxDotProduct = dotProduct;
                }
            }
            for (int k = 0; k < NUM_CLASSES; k++)
            {
                z += expf(dotProductV[k] - maxDotProduct);
            }
            heldLogLikelihood += dotProductV[heldLabels[j]] - (maxDotProduct + logf(z));
        }
        fprintf(outputFile,"Heldout: LogLikelihood: %f\n", heldLogLikelihood);
       
        
        //Check Regularization
        if (debugReg)
        {
            float totalBeta = 0.0f;
            for (int k = 0; k < NUM_CLASSES; k++)
            {
                for (int j = 0; j < FEATURE_COLS + 1; j++)
                {
                    totalBeta += beta[k][j];
                }
            }
            fprintf(outputFile,"Total beta: %f", totalBeta);
        }
        
        //Now write to a file named for the epoch etc.
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/beta-%d-%d-%f-%f.out", currentDirectory, e, NUM_FILES, regParam, initialLR);
        pFile = fopen(fName,"wb");
        for (int k = 0; k < NUM_CLASSES; k++)
        {
            fwrite(beta[k],sizeof(float),FEATURE_COLS + 1,pFile);
        }
        fclose(pFile);
        fflush(outputFile);
    }
    
    free(labels);
    free(features);
    free(heldLabels);
    free(heldFeatures);
    return 0;
}
