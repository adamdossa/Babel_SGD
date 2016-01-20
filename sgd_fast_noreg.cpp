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
#define NUM_EPOCHS 5
//#define INITIAL_LR 0.1
//#define ANNEAL_RATE 0.05

int main(int argc, const char * argv[])
{
    FILE * pFile;

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
    for (int i = 0; i < NUM_FILES * ROWS_PER_FILE; i++)
    {
        //printf("Label number: %d has value: %d\n",i,labels[i]);
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
    for (int i = 0; i<NUM_FILES * ROWS_PER_FILE; i++)
    {
        for (int j = 0; j<(FEATURE_COLS + 1); j++)
        {
            //printf("Feature coordinate: %d, %d has value: %f\n",i,j,features[i][j]);
        }
    }
    
    //Now run SGD
    printf("Running SGD\n");
    
    //Initialize parameters
    float ** beta = (float **) malloc(sizeof(float *) * NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        beta[i] = (float *) malloc(sizeof(float) * (FEATURE_COLS + 1));
        for (int j = 0; j < (FEATURE_COLS + 1); j++)
        {
            beta[i][j] = 0.0f;
        }
    }
    
//    time_t t0, t1;
//    clock_t c0, c1;
//    t0 = time(NULL);
//    c0 = clock();
    for (int e = 0; e < NUM_EPOCHS; e++)
    {
        printf("SGD Epoch: %d\n", e);
//        float ir = (float) INITIAL_LR;
//        float ar = (float) ANNEAL_RATE;
        float learningRate = 0.01f;//0.1f/(1 + (e/0.05f));
        for (int j = 0; j < NUM_FILES * ROWS_PER_FILE; j++)
        {
            //printf("Iterating on example: %d\n",j);
//            float batchCounter = ((float) j) / 1000.0f;
//            double integral;
//            double fractional = modf(batchCounter, &integral);
//            if (fractional == 0.0f) {
//                printf("Loop: %d\n",j);
//                t1 = time(NULL);
//                c1 = clock();
//                printf("Elapsed wall clock time: %ld\n", (long) (t1 - t0));
//                printf ("Elapsed CPU time: %f\n", (float) (c1 - c0)/CLOCKS_PER_SEC);
//                t0 = time(NULL);
//                c0 = clock();
//            }
            float z = 1.0f;
            float p_c_x[NUM_CLASSES - 1];
            for (int k = 0; k < (NUM_CLASSES - 1); k++)
            {
                float dotProduct = 0.0f;
                for (int d = 0; d < (FEATURE_COLS + 1); d++)
                {
                    dotProduct += beta[k][d] * features[j][d];
                }
                dotProduct = expf(dotProduct);
                z += dotProduct;
                p_c_x[k] = dotProduct;
            }
            for (int c = 0; c < NUM_CLASSES - 1; c++)
            {
//                float p_c_x = 0.0f;
//                for (int d = 0; d < (FEATURE_COLS + 1); d++)
//                {
//                    p_c_x += beta[c][d] * features[j][d];
//                }
//                p_c_x = expf(p_c_x) / z;
                float p_c_x_c = p_c_x[c] / z;
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
        printf("Calculating Likelihood\n");
        float logLikelihood = 0.0f;
        for (int j = 0; j < NUM_FILES * ROWS_PER_FILE; j++)
        {
            float z = 0.0f;
            float p_c_x = 0.0f;
            for (int k = 0; k < (NUM_CLASSES - 0); k++)
            {
                float dotProduct = 0.0f;
                for (int d = 0; d < (FEATURE_COLS + 1); d++)
                {
                    dotProduct += beta[k][d] * features[j][d];
                }
                dotProduct = expf(dotProduct);
                z += dotProduct;
                if (k == labels[j]) {
                    p_c_x = dotProduct;
                }
            }
            logLikelihood += logf(p_c_x/z);
            //printf("LogLikelihood: %f\n", logLikelihood);
        }
        printf("LogLikelihood: %f\n", logLikelihood);
    }
    return 0;
}

