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
#include <netinet/in.h>

#define FEATURE_COLS 360
#define NUM_FILES 1
#define NUM_HELDOUT_FILES 0
//#define NUM_FILES 1
//#define NUM_HELDOUT_FILES 4189
#define ROWS_PER_FILE 250
#define NUM_CLASSES 1000
#define NUM_EPOCHS 5
#define INITIAL_LR 0.1f
#define REG_PARAM 0.0f


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
    outputFile = fopen("/Users/AdamDossa/Documents/XCode/Babel_SGD/Babel_SGD/log.txt","w");
    
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
        outputFile = fopen(argv[3],"w");
    }
    
    //First read in the labels
    fprintf(outputFile, "Reading in labels\n");
    int * labels = (int *) malloc(sizeof(int) * NUM_FILES * ROWS_PER_FILE);
    for (int i = 0; i<NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/labels/train.%d.lab", rootDirectory, i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        n = ntohl(n);
        fread(&labels[i*ROWS_PER_FILE],4,n,pFile);
        for (int j = 0; j < n; j++)
        {
            labels[(i*ROWS_PER_FILE) + j] = ntohl(labels[(i*ROWS_PER_FILE) + j]);
        }
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i < NUM_FILES * ROWS_PER_FILE; i++)
        {
            fprintf(outputFile,"Train: Label number: %d has value: %d\n",i,labels[i]);
        }
    }
    fflush(outputFile);
    
    //Now read in the features - hold these in an array of arrays
    fprintf(outputFile,"Reading in features\n");
    float ** features = (float **) malloc(sizeof(float *) * NUM_FILES * ROWS_PER_FILE);
    for (int i = 0; i<NUM_FILES; i++) {
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
            features[(i * ROWS_PER_FILE) + j] = (float *) malloc(sizeof(float) * (m + 1));
            fread(features[(i * ROWS_PER_FILE) + j],sizeof(float),m,pFile);
            for (int k = 0; k < m; k++)
            {
                features[(i * ROWS_PER_FILE) + j][k] = bin2flt(&features[(i * ROWS_PER_FILE) + j][k]);
            }
            features[(i * ROWS_PER_FILE) + j][360] = 1.0f;
        }
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i<NUM_FILES * ROWS_PER_FILE; i++)
        {
            for (int j = 0; j<(FEATURE_COLS + 1); j++)
            {
                fprintf(outputFile,"Train: Feature coordinate: %d, %d has value: %f\n",i,j,features[i][j]);
            }
        }
    }
    fflush(outputFile);
    
    //Now do the same for the heldout data
    //First read in the held out labels
    fprintf(outputFile,"Reading in heldout labels\n");
    int * heldLabels = (int *) malloc(sizeof(int) * NUM_HELDOUT_FILES * ROWS_PER_FILE);
    for (int i = 0; i<NUM_HELDOUT_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/labels/heldout.%d.lab", rootDirectory, i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        n = ntohl(n);
        fread(&heldLabels[i*ROWS_PER_FILE],4,n,pFile);
        for (int j = 0; j < n; j++)
        {
            heldLabels[(i*ROWS_PER_FILE) + j] = ntohl(heldLabels[(i*ROWS_PER_FILE) + j]);
        }
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i < NUM_HELDOUT_FILES * ROWS_PER_FILE; i++)
        {
            fprintf(outputFile,"Heldout: Label number: %d has value: %d\n",i,heldLabels[i]);
        }
    }
    fflush(outputFile);
    
    //Now read in the features - hold these in an array of arrays
    fprintf(outputFile,"Reading in heldout features\n");
    float ** heldFeatures = (float **) malloc(sizeof(float *) * NUM_HELDOUT_FILES * ROWS_PER_FILE);
    for (int i = 0; i<NUM_HELDOUT_FILES; i++) {
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
            heldFeatures[(i * ROWS_PER_FILE) + j] = (float *) malloc(sizeof(float) * (m + 1));
            fread(heldFeatures[(i * ROWS_PER_FILE) + j],sizeof(float),m,pFile);
            for (int k = 0; k < m; k++)
            {
                heldFeatures[(i * ROWS_PER_FILE) + j][k] = bin2flt(&heldFeatures[(i * ROWS_PER_FILE) + j][k]);
            }
            heldFeatures[(i * ROWS_PER_FILE) + j][360] = 1.0f;
        }
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i<NUM_HELDOUT_FILES * ROWS_PER_FILE; i++)
        {
            for (int j = 0; j<(FEATURE_COLS + 1); j++)
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
    int noOfTrainingExamples = (NUM_FILES * ROWS_PER_FILE);
    //Start looping for NUM_EPOCHS
    for (int e = 0; e < NUM_EPOCHS; e++)
    {
        fprintf(outputFile,"SGD Epoch: %d\n", e);
        fflush(outputFile);
        //Calculate learning rate for epoch
        //float learningRate = INITIAL_LR/(1 + e/ANNEAL_RATE);
        //Loop over examples
        float epochExamples = (e * noOfTrainingExamples);
        for (int j = 0; j < noOfTrainingExamples; j++)
        {
            float learningRate = INITIAL_LR/(1 + ((epochExamples + j)/(float) noOfTrainingExamples));
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
                        beta[c][d] += learningRate*(features[j][d]*(1.0f - p_c_x_c) - (2 * REG_PARAM * beta[c][d]));
                    }
                } else {
                    for (int d = 0; d < (FEATURE_COLS + 1); d++)
                    {
                        beta[c][d] += learningRate*(features[j][d]*(0.0f - p_c_x_c) - (2 * REG_PARAM * beta[c][d]));
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
        for (int j = 0; j < NUM_FILES * ROWS_PER_FILE; j++)
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
            logLikelihood += dotProductV[labels[j]] - (maxDotProduct + logf(z)) - (REG_PARAM * regTerm);
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
        for (int j = 0; j < NUM_HELDOUT_FILES * ROWS_PER_FILE; j++)
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
        snprintf(fName, sizeof(char) * 200,"%s/beta.%d.%d.%f.%f", currentDirectory, e, NUM_FILES, REG_PARAM, INITIAL_LR);
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