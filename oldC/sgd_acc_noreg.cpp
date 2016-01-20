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

#define FEATURE_COLS 360
#define NUM_FILES 10001
#define ROWS_PER_FILE 250
#define NUM_CLASSES 1000
#define NUM_EPOCHS 10
#define INITIAL_LR 0.01f
#define ANNEAL_RATE 10.0f

int main(int argc, const char * argv[])
{
    FILE * pFile;
    bool debug = false;
    bool timeLoops = false;
    //First read in the labels
    printf("Reading in labels\n");
    int * labels = (int *) malloc(sizeof(int) * NUM_FILES * ROWS_PER_FILE);
    for (int i = 0; i<NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"/Users/AdamDossa/Documents/XCode/train/train.%d.lab.le", i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        fread(&labels[i*ROWS_PER_FILE],4,n,pFile);
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i < NUM_FILES * ROWS_PER_FILE; i++)
        {
            printf("Label number: %d has value: %d\n",i,labels[i]);
        }
    }
    //Now read in the features - hold these in an array of arrays
    printf("Reading in features\n");
    float ** features = (float **) malloc(sizeof(float *) * NUM_FILES * ROWS_PER_FILE);
    for (int i = 0; i<NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"/Users/AdamDossa/Documents/XCode/train/train.%d.fea.le", i);
        pFile = fopen(fName,"rb");
        int n;
        fread(&n,4,1,pFile);
        for (int j = 0; j < n; j++)
        {
            int m;
            fread(&m,4,1,pFile);
            features[(i * ROWS_PER_FILE) + j] = (float *) malloc(sizeof(float) * (m + 1));
            fread(features[(i * ROWS_PER_FILE) + j],sizeof(float),m,pFile);
            features[(i * ROWS_PER_FILE) + j][360] = 1.0f;
        }
        fclose(pFile);
    }
    if (debug) {
        for (int i = 0; i<NUM_FILES * ROWS_PER_FILE; i++)
        {
            for (int j = 0; j<(FEATURE_COLS + 1); j++)
            {
                printf("Feature coordinate: %d, %d has value: %f\n",i,j,features[i][j]);
            }
        }
    }
    //Now run SGD
    printf("Running SGD\n");
    
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
    
    //Start looping for NUM_EPOCHS
    for (int e = 0; e < NUM_EPOCHS; e++)
    {
        printf("SGD Epoch: %d\n", e);
        //Calculate learning rate for epoch
        float learningRate = INITIAL_LR/(1 + e/ANNEAL_RATE);
        //Loop over examples
        for (int j = 0; j < NUM_FILES * ROWS_PER_FILE; j++)
        {
            if (debug) printf("Train: Iterating on example: %d\n",j);
            //Generate loop times (every 1000 loops)
            if (timeLoops) {
                float batchCounter = ((float) j) / 1000.0f;
                double integral;
                double fractional = modf(batchCounter, &integral);
                if (fractional == 0.0f) {
                    printf("Train: Loop: %d\n",j);
                    t1 = time(NULL);
                    c1 = clock();
                    printf("Train: Elapsed wall clock time: %ld\n", (long) (t1 - t0));
                    printf ("Train: Elapsed CPU time: %f\n", (float) (c1 - c0)/CLOCKS_PER_SEC);
                    t0 = time(NULL);
                    c0 = clock();
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
                //Update rule depends on indicator function
                if (c == labels[j])
                {
                    for (int d = 0; d < (FEATURE_COLS + 1); d++)
                    {
                        beta[c][d] += learningRate*features[j][d]*(1.0f - p_c_x_c);
                    }
                } else {
                    for (int d = 0; d < (FEATURE_COLS + 1); d++)
                    {
                        beta[c][d] += learningRate*features[j][d]*(0.0f - p_c_x_c);
                    }                    
                }
            }
        }
        
        //Now do the log-likelihood calculation (on training data)
        printf("Calculating Likelihood\n");
        //Initialize logLikelihood to 0
        float logLikelihood = 0.0f;
        if (timeLoops) {
            t0 = time(NULL);
            c0 = clock();
        }
        //Loop over each example
        for (int j = 0; j < NUM_FILES * ROWS_PER_FILE; j++)
        {
            if (debug) printf("Likelihood: Iterating on example: %d\n",j);
            if (timeLoops) {
                float batchCounter = ((float) j) / 1000.0f;
                double integral;
                double fractional = modf(batchCounter, &integral);
                if (fractional == 0.0f) {
                    printf("Likelihood Loop: %d\n",j);
                    t1 = time(NULL);
                    c1 = clock();
                    printf("Likelihood: Elapsed wall clock time: %ld\n", (long) (t1 - t0));
                    printf ("Likelihood: Elapsed CPU time: %f\n", (float) (c1 - c0)/CLOCKS_PER_SEC);
                    t0 = time(NULL);
                    c0 = clock();
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
            logLikelihood += dotProductV[labels[j]] - (maxDotProduct + logf(z));
        }
        printf("LogLikelihood: %f\n", logLikelihood);
        //Now write to a file named for the epoch
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"/Users/AdamDossa/Documents/XCode/Babel_SGD/Babel_SGD/beta.%d", e);
        pFile = fopen(fName,"wb");
        for (int k = 0; k < NUM_CLASSES; k++)
        {
            fwrite(beta[k],sizeof(float),361,pFile);
        }
        fclose(pFile);
    }
    return 0;
}

