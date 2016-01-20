//
//  main.cpp
//  Babel_SGD
//
//  Created by Adam Dossa on 11/06/2013.
//  Copyright (c) 2013 Adam Dossa. All rights reserved.
//
//  Compute train & heldout log likelihood based on beta* files output by SGD / bound methods
//  In this case we use the beta* files output by main_sgd_42 & main_sgd_sub to calculate overall likelihood

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
//#include <libc.h>
#include <netinet/in.h>
#include <set>

#define FEATURE_COLS 360
#define RANK 360
//#define NUM_FILES 1
//#define NUM_HELDOUT_FILES 100
#define NUM_FILES 30700
#define NUM_HELDOUT_FILES 4190
#define NUM_CLASSES_LARGE 1000
#define NUM_CLASSES_SMALL 42
#define START_EPOCH 0
//We assume same parameters here are used across all small clases
//Initial learning rate (can be set via command line argument)
#define INITIAL_LR_LARGE 0.1f //e.g. used to goto 42 mapping (e.g. main_sgd_42)
#define INITIAL_LR_SMALL 0.001f //e.g. used to goto sub mapping (e.g. main_sgd_sub)
//Regularization parameter (can be set via command line argument)
#define REG_PARAM_LARGE 10.0f
#define REG_PARAM_SMALL 10.0f

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
    bool doTrainLikelihood = false;
    bool doTestLikelihood = true;
    
    const char * heldFileName = "heldout";
    const char * trainFileName = "train";
    const char * labelDir = "labels";
    const char * featuresDir = "features";
    const char * mappingDir = "labels_42";

    //Defaults (for my mac)
    const char * rootDirectory = "/Users/AdamDossa/Documents/Columbia/GRA/Babel/";
    const char * currentDirectory = "/Users/AdamDossa/Documents/XCode/Babel_SGD/Babel_SGD/log/";
    
    float regParamLarge = REG_PARAM_LARGE;
    float regParamSmall = REG_PARAM_SMALL;
    float initialLRLarge = INITIAL_LR_LARGE;
    float initialLRSmall = INITIAL_LR_SMALL;
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
    if (argc > 3)
    {
        regParamLarge = (float) atof(argv[3]);
    }
    if (argc > 4)
    {
        initialLRLarge = (float) atof(argv[4]);
    }
    if (argc > 5)
    {
        regParamSmall = (float) atof(argv[5]);
    }
    if (argc > 6)
    {
        initialLRSmall = (float) atof(argv[6]);
    }
    if (argc > 7)
    {
        rank = (int) atoi(argv[7]);
    }
    
    //We want to find an unused log file name (to some limit)
    char logFileName[200];
    for (int i = 0; i < 100; i++)
    {
        snprintf(logFileName, sizeof(char) * 200,"%s/log/log_sub_likelihood-%d-%d-%d-%f-%f-%f-%f.txt.%d", currentDirectory, rank, START_EPOCH, NUM_FILES, regParamLarge, initialLRLarge, regParamSmall, initialLRSmall, i);
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

    
    //Now read in the label mapping (from large classes to small classes)
    //and determine the set of original labels that map to our phone
    fprintf(outputFile, "Reading in label mappings\n");
    int * labelMapping = (int *) malloc(sizeof(int) * NUM_CLASSES_LARGE);
    char fName[200];
    int n;
    snprintf(fName, sizeof(char) * 200,"%s/%s/label_mapping.lab", rootDirectory, mappingDir);
    pFile = fopen(fName,"rb");
    fread(&n,4,1,pFile);
    //Ignore n and just read a correct number of mapping labels
    fread(labelMapping,4,NUM_CLASSES_LARGE,pFile);
    fclose(pFile);
    if (debug) {
        for (int i = 0; i < NUM_CLASSES_LARGE; i++)
        {
            fprintf(outputFile,"Label Mapping: %d has value: %d\n",i,labelMapping[i]);
        }
    }
    fflush(outputFile);
    
    //Create mappings down to individual phones
    std::set <int> phoneOrigLabels;
    int * subLabelMapping = (int *) malloc(sizeof(int) * NUM_CLASSES_LARGE);
    int * noClasses = (int *) malloc(sizeof(int) * NUM_CLASSES_SMALL);
    for (int j = 0; j < NUM_CLASSES_SMALL; j++)
    {
        for (int i = 0; i < NUM_CLASSES_LARGE; i++)
        {
            if (labelMapping[i]==j) {
                phoneOrigLabels.insert(i);
            }
        }
        noClasses[j] = phoneOrigLabels.size();
        std::set<int>::iterator phoneIter;
        int counter = 0;
        for (phoneIter = phoneOrigLabels.begin(); phoneIter!=phoneOrigLabels.end();phoneIter++)
        {
            subLabelMapping[*phoneIter] = counter;
            printf("%d\n",*phoneIter);
            counter++;
        }
        phoneOrigLabels.clear();
    }
    
    //First calculate the number of training examples (since not every file is guaranteed to have 250 rows)
    fprintf(outputFile, "Calculating number of training examples\n");
    int noOfTrainingExamples = 0;
    int readSoFar = 0;
    int * labels;
    float ** features;
    if (doTrainLikelihood) {
        for (int i = 0; i < NUM_FILES; i++) {
            char fName[200];
            snprintf(fName, sizeof(char) * 200,"%s/%s/%s.%d.lab", rootDirectory, labelDir, trainFileName, i);
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
        labels = (int *) malloc(sizeof(int) * noOfTrainingExamples);
        for (int i = 0; i < NUM_FILES; i++) {
            char fName[200];
            snprintf(fName, sizeof(char) * 200,"%s/%s/%s.%d.lab", rootDirectory, labelDir, trainFileName, i);
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
        features = (float **) malloc(sizeof(float *) * noOfTrainingExamples);
        for (int i = 0; i < NUM_FILES; i++) {
            char fName[200];
            snprintf(fName, sizeof(char) * 200,"%s/%s/%s.%d.fea", rootDirectory, featuresDir,trainFileName, i);
            pFile = fopen(fName,"rb");
            int n;
            fread(&n,4,1,pFile);
            n = ntohl(n);
            for (int j = 0; j < n; j++)
            {
                int m, m_read;
                fread(&m,4,1,pFile);
                m = ntohl(m);
                m_read = m;
                if (m_read > rank)
                {
                    m_read = rank;
                }
                features[readSoFar + j] = (float *) malloc(sizeof(float) * (m_read + 1));
                fread(features[readSoFar + j],sizeof(float),m_read,pFile);
                fseek(pFile, (m - m_read) * sizeof(float), SEEK_CUR);
                for (int k = 0; k < m_read; k++)
                {
                    features[readSoFar + j][k] = bin2flt(&features[readSoFar + j][k]);
                }
                features[readSoFar + j][m_read] = 1.0f;
            }
            readSoFar += n;
            fclose(pFile);
        }
        if (debug) {
            for (int i = 0; i < noOfTrainingExamples; i++)
            {
                for (int j = 0; j<(rank + 1); j++)
                {
                    fprintf(outputFile,"Train: Feature coordinate: %d, %d has value: %f\n",i,j,features[i][j]);
                }
            }
        }
        fflush(outputFile);
    }
    
    //Now do the same for the heldout data
    //First calculate the number of training examples (since not every file is guaranteed to have 250 rows)
    int noOfHeldoutExamples = 0;
    int * heldLabels;
    float ** heldFeatures;
    if (doTestLikelihood) {
        fprintf(outputFile, "Calculating number of heldout examples\n");
        for (int i = 0; i < NUM_HELDOUT_FILES; i++) {
            char fName[200];
            snprintf(fName, sizeof(char) * 200,"%s/%s/%s.%d.lab", rootDirectory, labelDir, heldFileName, i);
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
        heldLabels = (int *) malloc(sizeof(int) * noOfHeldoutExamples);
        for (int i = 0; i<NUM_HELDOUT_FILES; i++) {
            char fName[200];
            snprintf(fName, sizeof(char) * 200,"%s/%s/%s.%d.lab", rootDirectory, labelDir, heldFileName, i);
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
        heldFeatures = (float **) malloc(sizeof(float *) * noOfHeldoutExamples);
        for (int i = 0; i < NUM_HELDOUT_FILES; i++) {
            char fName[200];
            snprintf(fName, sizeof(char) * 200,"%s/%s/%s.%d.fea", rootDirectory, featuresDir, heldFileName, i);
            pFile = fopen(fName,"rb");
            int n;
            fread(&n,4,1,pFile);
            n = ntohl(n);
            for (int j = 0; j < n; j++)
            {
                int m, m_read;
                fread(&m,4,1,pFile);
                m = ntohl(m);
                m_read = m;
                if (m_read > rank)
                {
                    m_read = rank;
                }
                heldFeatures[readSoFar + j] = (float *) malloc(sizeof(float) * (m_read + 1));
                fread(heldFeatures[readSoFar + j],sizeof(float),m_read,pFile);
                fseek(pFile, (m - m_read) * sizeof(float), SEEK_CUR);
                for (int k = 0; k < m_read; k++)
                {
                    heldFeatures[readSoFar + j][k] = bin2flt(&heldFeatures[readSoFar + j][k]);
                }
                heldFeatures[readSoFar + j][m_read] = 1.0f;
            }
            readSoFar += n;
            fclose(pFile);
        }
        if (debug) {
            for (int i = 0; i < noOfHeldoutExamples; i++)
            {
                for (int j = 0; j < (rank + 1); j++)
                {
                    fprintf(outputFile,"Heldout: Feature coordinate: %d, %d has value: %f\n",i,j,heldFeatures[i][j]);
                }
            }
        }
        fflush(outputFile);
    }
    
    //Initialize beta parameters, one for each large class and small class
    float ** beta = (float **) malloc(sizeof(float *) * NUM_CLASSES_SMALL);
    for (int i = 0; i < NUM_CLASSES_SMALL; i++)
    {
        beta[i] = (float *) malloc(sizeof(float) * (rank + 1));
        for (int j = 0; j < (rank + 1); j++)
        {
            beta[i][j] = 0.0f;
        }
    }

    //Now initialize beta parameters, one for each small class
    float *** betaMatrix = (float ***) malloc(sizeof(float **) * NUM_CLASSES_SMALL);
    for (int i = 0; i < NUM_CLASSES_SMALL; i++)
    {
        betaMatrix[i] = (float **) malloc(sizeof(float *) * noClasses[i]);
        for (int k = 0; k < noClasses[i]; k++)
        {
            betaMatrix[i][k] = (float *) malloc(sizeof(float) * (rank +1));
            for (int j = 0; j < (rank + 1); j++)
            {
                betaMatrix[i][k][j] = 0.0f;
            }
        }
    }
    
    //Check for any beta* files of appropriate name, and initialize using latest epoch
    int startingEpoch = START_EPOCH;
        
    for (int e = startingEpoch; e < 99; e++)
    {
        fprintf(outputFile,"SGD Epoch: %d\n", e);
        
        char lsName[200];
        FILE *lsFile;
        
        //Now read in beta files as needed
        //Initialize using beta file
        snprintf(lsName, sizeof(char) * 200,"%s/beta/beta_sgd_42-%d-%d-%d-%f-%f.out", currentDirectory, rank, e, NUM_FILES, regParamLarge, initialLRLarge);
        fprintf(outputFile,"Initializing from file: %s\n", lsName);
        lsFile = fopen(lsName,"rb");
        if (!lsFile) {
            fprintf(outputFile, "No file found - finishing\n");
            exit(0);
        }
        for (int k = 0; k < NUM_CLASSES_SMALL; k++)
        {
            fread(beta[k],sizeof(float),rank + 1, lsFile);
        }
        fclose(lsFile);
        fflush(outputFile);
        
        for (int i = 0; i < NUM_CLASSES_SMALL; i++)
        {
            snprintf(lsName, sizeof(char) * 200,"%s/beta/beta_sub-%d-%d-%d-%d-%f-%f.out", currentDirectory, i, rank, e, NUM_FILES, regParamSmall, initialLRSmall);
            if (debug) {
                fprintf(outputFile,"Initializing from sub-file: %s\n", lsName);
            }
            lsFile = fopen(lsName,"rb");
            if (!lsFile) {
                fprintf(outputFile, "No file found - finishing\n");
                exit(0);
            }
            for (int k = 0; k < noClasses[i]; k++)
            {
                fread(betaMatrix[i][k],sizeof(float),rank + 1, lsFile);
            }
            fclose(lsFile);
            fflush(outputFile);            
        }
        
        if (doTrainLikelihood) {
            //Now do the log-likelihood calculation (on training data)
            fprintf(outputFile,"Train: Calculating Likelihood\n");
            fflush(outputFile);
            
            //Initialize logLikelihood to 0
            float logLikelihood = 0.0f;
            
            //Loop over each example
            for (int j = 0; j < noOfTrainingExamples; j++)
            {
                if (debug) fprintf(outputFile,"Train Likelihood: Iterating on example: %d\n",j);
                float z = 0.0f;
                float dotProductV[NUM_CLASSES_SMALL];
                float maxDotProduct;
                for (int k = 0; k < NUM_CLASSES_SMALL; k++)
                {
                    float dotProduct = 0.0f;
                    for (int d = 0; d < (rank + 1); d++)
                    {
                        dotProduct += beta[k][d] * features[j][d];
                    }
                    dotProductV[k] = dotProduct;
                    if (k==0) maxDotProduct = dotProduct;
                    if (maxDotProduct < dotProduct) {
                        maxDotProduct = dotProduct;
                    }
                }
                for (int k = 0; k < NUM_CLASSES_SMALL; k++)
                {
                    z += expf(dotProductV[k] - maxDotProduct);
                }
                float regTerm = 0.0f;
                for (int d = 0; d < (rank + 1); d++)
                {
                    regTerm += powf(beta[labelMapping[labels[j]]][d],2);
                }
                float firstProb = dotProductV[labelMapping[labels[j]]] - (maxDotProduct + logf(z));
                float firstReg = ((0.5f * regParamLarge * regTerm)/(float) noOfTrainingExamples);
                //logLikelihood += dotProductV[labelMapping[labels[j]]] - (maxDotProduct + logf(z)) - ((0.5f * regParamLarge * regTerm)/(float) noOfTrainingExamples);
                z = 0.0f;
                int subClass = labelMapping[labels[j]]; //From 0 - 42
                int subLabel = subLabelMapping[labels[j]];//From 0 - noClasses[subClass]
                float dotProductSubV[noClasses[subClass]];
                for (int k = 0; k < noClasses[subClass]; k++)
                {
                    float dotProduct = 0.0f;
                    for (int d = 0; d < (rank + 1); d++)
                    {
                        dotProduct += betaMatrix[subClass][k][d] * features[j][d];
                    }
                    dotProductSubV[k] = dotProduct;
                    if (k==0) maxDotProduct = dotProduct;
                    if (maxDotProduct < dotProduct) {
                        maxDotProduct = dotProduct;
                    }
                }
                for (int k = 0; k < noClasses[subClass]; k++)
                {
                    z += expf(dotProductSubV[k] - maxDotProduct);
                }
                regTerm = 0.0f;
                for (int d = 0; d < (rank + 1); d++)
                {
                    regTerm += powf(betaMatrix[subClass][subLabel][d],2);
                }
                float secondProb = dotProductSubV[subLabel] - (maxDotProduct + logf(z));
                float secondReg = ((0.5f * regParamSmall * regTerm)/(float) noOfTrainingExamples);
                logLikelihood += (firstProb + secondProb) - firstReg - secondReg;
            }
            fprintf(outputFile,"Train: LogLikelihood: %f\n", logLikelihood);
        }
        
        if (doTestLikelihood) {
            //Now do the log-likelihood calculation on held out data
            fprintf(outputFile,"Heldout: Calculating Likelihood\n");
            fflush(outputFile);
            //Initialize logLikelihood to 0
            float heldLogLikelihood = 0.0f;
            
            //Loop over each example
            for (int j = 0; j < noOfHeldoutExamples; j++)
            {
                if (debug) fprintf(outputFile,"Heldout Likelihood - Iterating on example: %d\n",j);

                float z = 0.0f;
                float dotProductV[NUM_CLASSES_SMALL];
                float maxDotProduct;
                for (int k = 0; k < NUM_CLASSES_SMALL; k++)
                {
                    float dotProduct = 0.0f;
                    for (int d = 0; d < (rank + 1); d++)
                    {
                        dotProduct += beta[k][d] * heldFeatures[j][d];
                    }
                    dotProductV[k] = dotProduct;
                    if (k==0) maxDotProduct = dotProduct;
                    if (maxDotProduct < dotProduct) {
                        maxDotProduct = dotProduct;
                    }
                }
                for (int k = 0; k < NUM_CLASSES_SMALL; k++)
                {
                    z += expf(dotProductV[k] - maxDotProduct);
                }
                float regTerm = 0.0f;
                for (int d = 0; d < (rank + 1); d++)
                {
                    regTerm += powf(beta[labelMapping[heldLabels[j]]][d],2);
                }
                float firstProb = dotProductV[labelMapping[heldLabels[j]]] - (maxDotProduct + logf(z));
                //logLikelihood += dotProductV[labelMapping[labels[j]]] - (maxDotProduct + logf(z)) - ((0.5f * regParamLarge * regTerm)/(float) noOfTrainingExamples);
                z = 0.0f;
                int subClass = labelMapping[heldLabels[j]]; //From 0 - 42
                int subLabel = subLabelMapping[heldLabels[j]];//From 0 - noClasses[subClass]
                float dotProductSubV[noClasses[subClass]];
                for (int k = 0; k < noClasses[subClass]; k++)
                {
                    float dotProduct = 0.0f;
                    for (int d = 0; d < (rank + 1); d++)
                    {
                        dotProduct += betaMatrix[subClass][k][d] * heldFeatures[j][d];
                    }
                    dotProductSubV[k] = dotProduct;
                    if (k==0) maxDotProduct = dotProduct;
                    if (maxDotProduct < dotProduct) {
                        maxDotProduct = dotProduct;
                    }
                }
                for (int k = 0; k < noClasses[subClass]; k++)
                {
                    z += expf(dotProductSubV[k] - maxDotProduct);
                }
                regTerm = 0.0f;
                for (int d = 0; d < (rank + 1); d++)
                {
                    regTerm += powf(betaMatrix[subClass][subLabel][d],2);
                }
                float secondProb = dotProductSubV[subLabel] - (maxDotProduct + logf(z));
                heldLogLikelihood += (firstProb + secondProb);
            }
            fprintf(outputFile,"Heldout: LogLikelihood: %f\n", heldLogLikelihood);
        }
        
        fflush(outputFile);
    }
    
    free(labels);
    free(features);
    free(heldLabels);
    free(heldFeatures);
    return 0;
}
