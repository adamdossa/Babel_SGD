//
//  main.cpp
//  Babel_SGD
//
//  Created by Adam Dossa on 11/06/2013.
//  Copyright (c) 2013 Adam Dossa. All rights reserved.
//
//  Runs Stochastic Gradient Descent on Babel training set
//  Calculate parameters for an individual phone (one of 42)

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
//#include <libc.h>
#include <set>
#include <netinet/in.h>

//The number of features in the input data
#define FEATURE_COLS 360
//The number of training input files to read in (each file has approx 250 examples)
//#define NUM_FILES 10
//#define NUM_HELDOUT_FILES 10
#define NUM_FILES 30700
#define NUM_HELDOUT_FILES 4190
//The number of possible phone classes
#define NUM_CLASSES 1000
//Run for this number of epochs
#define NUM_EPOCHS 100
//Initial learning rate (can be set via command line argument)
#define INITIAL_LR 0.001f
//Regularization parameter (can be set via command line argument)
#define REG_PARAM 10.0f
//Phone label to run SGD sub problem for
#define PHONE 1

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
    bool doTrainLikelihood = true;
    bool doTestLikelihood = true;
    
    const char * heldFileName = "heldout";
    const char * trainFileName = "train";
    const char * labelDir = "labels";
    const char * featuresDir = "features";
    const char * mappingDir = "labels_42";

    
    //Defaults (for my mac)
    const char * rootDirectory = "/Users/AdamDossa/Documents/Columbia/GRA/Babel/";
    const char * currentDirectory = "/Users/AdamDossa/Documents/XCode/Babel_SGD/Babel_SGD/log/";
    float regParam = REG_PARAM;
    float initialLR = INITIAL_LR;
    int rank = FEATURE_COLS;
    int phone = PHONE;
    
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
    if (argc > 5)
    {
        phone = (int) atoi(argv[5]);
    }
    
    //We want to find an unused log file name (to some limit)
    char logFileName[200];
    for (int i = 0; i < 100; i++)
    {
        snprintf(logFileName, sizeof(char) * 200,"%s/log/log_sgd_sub-%d-%d-%d-%f-%f-%d.txt.%d", currentDirectory, rank, NUM_EPOCHS, NUM_FILES, regParam, initialLR, phone, i);
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
    int * labelMapping = (int *) malloc(sizeof(int) * NUM_CLASSES);
    char fName[200];
    int n;
    snprintf(fName, sizeof(char) * 200,"%s/%s/label_mapping.lab", rootDirectory, mappingDir);
    pFile = fopen(fName,"rb");
    fread(&n,4,1,pFile);
    //Ignore n and just read a correct number of mapping labels
    fread(labelMapping,4,NUM_CLASSES,pFile);
    fclose(pFile);
    if (debug) {
        for (int i = 0; i < NUM_CLASSES; i++)
        {
            fprintf(outputFile,"Label Mapping: %d has value: %d\n",i,labelMapping[i]);
        }
    }
    fflush(outputFile);
    std::set <int> phoneOrigLabels;
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        if (labelMapping[i]==phone) {
            phoneOrigLabels.insert(i);
        }
    }
    
    int noLabels = phoneOrigLabels.size();
    
    //First calculate the number of training examples (since not every file is guaranteed to have 250 rows)
    fprintf(outputFile, "Calculating number of training examples\n");
    int noOfTrainingExamples = 0;
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
    int readSoFar = 0;
    int phoneCount = 0; //Number of examples that match our phone

    //Initially we set to larger size, and restrict later
    int * labels = (int *) malloc(sizeof(int) * noOfTrainingExamples);
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
            int tempLabel = ntohl(labels[readSoFar + j]);
            if (labelMapping[tempLabel] == phone) {
                phoneCount++;
            }
            labels[readSoFar + j] = tempLabel;
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
    fprintf(outputFile, "Number of matched examples: %d\n", phoneCount);
    fflush(outputFile);
    
    //Now read in the features - hold these in an array of arrays - assume same number as training examples
    fprintf(outputFile,"Reading in features\n");
    readSoFar = 0;
    float ** features = (float **) malloc(sizeof(float *) * noOfTrainingExamples);
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
    
    //Now restrict to just the set we are interested in
    //First create mapping from real labels, to reindexed labels
    int * subLabelMapping = (int *) malloc(sizeof(int) * NUM_CLASSES);
    std::set<int>::iterator phoneIter;
    fprintf(outputFile, "Number of original labels being used: %d\n", noLabels);
    int counter = 0;
    for (phoneIter = phoneOrigLabels.begin(); phoneIter!=phoneOrigLabels.end();phoneIter++)
    {
        subLabelMapping[*phoneIter] = counter;
        printf("%d\n",*phoneIter);
        counter++;
    }
    
    int * subLabels = (int *) malloc(sizeof(int) * phoneCount);
    float ** subFeatures = (float **) malloc(sizeof(float *) * phoneCount);
    counter = 0;
    for (int i = 0; i < noOfTrainingExamples; i++)
    {
        if (labelMapping[labels[i]]==phone) {
            subLabels[counter] = subLabelMapping[labels[i]];
            subFeatures[counter] = (float *) malloc(sizeof(float) * (rank + 1));
            for (int j = 0; j < rank + 1; j++)
            {
                subFeatures[counter][j] = features[i][j];
            }
            counter++;
        }
    }
    
    noOfTrainingExamples = phoneCount;
    
    //Kill off the previous labels / features
    free(labels);
    free(features);
    
    //Now read in the test data
    //First calculate the number of training examples (since not every file is guaranteed to have 250 rows)
    int noOfHeldoutExamples = 0;
    int * heldLabels;
    float ** heldFeatures;
    int * subHeldLabels;
    float ** subHeldFeatures;
    
    int heldPhoneCount = 0;
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
                int tempLabel = ntohl(heldLabels[readSoFar + j]);
                if (labelMapping[tempLabel] == phone) {
                    heldPhoneCount++;
                }
                heldLabels[readSoFar + j] = tempLabel;
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
        fprintf(outputFile, "Number of matched heldout examples: %d\n", heldPhoneCount);

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
        subHeldLabels = (int *) malloc(sizeof(int) * heldPhoneCount);
        subHeldFeatures = (float **) malloc(sizeof(float *) * heldPhoneCount);
        counter = 0;
        for (int i = 0; i < noOfHeldoutExamples; i++)
        {
            if (labelMapping[heldLabels[i]]==phone) {
                subHeldLabels[counter] = subLabelMapping[heldLabels[i]];
                subHeldFeatures[counter] = (float *) malloc(sizeof(float) * (rank + 1));
                for (int j = 0; j < rank + 1; j++)
                {
                    subHeldFeatures[counter][j] = heldFeatures[i][j];
                }
                counter++;
            }
        }
        
        noOfHeldoutExamples = heldPhoneCount;

    }
    
    //Now run SGD
    fprintf(outputFile,"Running SGD\n");
    fflush(outputFile);
    
    //Initialize beta parameters, one for each class
    float ** beta = (float **) malloc(sizeof(float *) * noLabels);
    for (int i = 0; i < noLabels; i++)
    {
        beta[i] = (float *) malloc(sizeof(float) * (rank + 1));
        for (int j = 0; j < (rank + 1); j++)
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
    snprintf(lsName, sizeof(char) * 200,"ls %s/beta/beta_sub-%d-%d-%s-%d-%f-%f.out | grep 'beta' | cut -d'-' -f4", currentDirectory, phone, rank, "*", NUM_FILES, regParam, initialLR);
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
        if ((1 + atoi(lsBuffer)) > startingEpoch) {
            startingEpoch = 1 + atoi(lsBuffer);
        }
    }
    pclose(lsFile);
    
    //Now read in beta files as needed
    if (startingEpoch > 0)
    {
        //Initialize using beta file
        snprintf(lsName, sizeof(char) * 200,"%s/beta/beta_sub-%d-%d-%d-%d-%f-%f.out", currentDirectory, phone, rank, startingEpoch - 1, NUM_FILES, regParam, initialLR);
        fprintf(outputFile,"Initializing from file: %s\n", lsName);
        lsFile = fopen(lsName,"rb");
        for (int k = 0; k < noLabels; k++)
        {
            float t1 = beta[k][0];
            fread(beta[k],sizeof(float),rank + 1, lsFile);
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
            float dotProductV[noLabels];
            //In order to avoid overflows, we use the identity:
            //log(sum_i(exp(beta_i.x_i))) = m + log(sum_i(exp(beta_i.x_i - m)))
            //where m is the maxDotProduct
            float maxDotProduct;
            for (int k = 0; k < noLabels; k++)
            {
                float dotProduct = 0.0f;
                for (int d = 0; d < (rank + 1); d++)
                {
                    dotProduct += beta[k][d] * subFeatures[j][d];
                }
                dotProductV[k] = dotProduct;
                if (k==0) maxDotProduct = dotProduct;
                if (maxDotProduct < dotProduct) {
                    maxDotProduct = dotProduct;
                }
            }
            //Now calculate z as above
            for (int k = 0; k < noLabels; k++)
            {
                z += expf(dotProductV[k] - maxDotProduct);
            }
            z = maxDotProduct + logf(z);
            //Update each beta
            for (int c = 0; c < noLabels; c++)
            {
                //Calculate p(c_c|x_j, beta_c)
                float p_c_x_c = expf(dotProductV[c] - z);
                
                //Update rule depends on indicator function (Gaussian Prior)
                if (c == subLabels[j])
                {
                    for (int d = 0; d < (rank + 1); d++)
                    {
                        beta[c][d] += learningRate*(subFeatures[j][d]*(1.0f - p_c_x_c) - (regParam * beta[c][d])/noOfTrainingExamples);
                    }
                } else {
                    for (int d = 0; d < (rank + 1); d++)
                    {
                        beta[c][d] += learningRate*(subFeatures[j][d]*(0.0f - p_c_x_c) - (regParam * beta[c][d])/noOfTrainingExamples);
                    }
                }
            }
        }
        
        //Now write to a file named for the epoch etc.
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/beta/beta_sub-%d-%d-%d-%d-%f-%f.out", currentDirectory, phone, rank, e, NUM_FILES, regParam, initialLR);
        pFile = fopen(fName,"wb");
        for (int k = 0; k < noLabels; k++)
        {
            fwrite(beta[k],sizeof(float),rank + 1,pFile);
        }
        fclose(pFile);
        
        if (doTrainLikelihood) {
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
                float dotProductV[noLabels];
                float maxDotProduct;
                for (int k = 0; k < noLabels; k++)
                {
                    float dotProduct = 0.0f;
                    for (int d = 0; d < (rank + 1); d++)
                    {
                        dotProduct += beta[k][d] * subFeatures[j][d];
                    }
                    dotProductV[k] = dotProduct;
                    if (k==0) maxDotProduct = dotProduct;
                    if (maxDotProduct < dotProduct) {
                        maxDotProduct = dotProduct;
                    }
                }
                for (int k = 0; k < noLabels; k++)
                {
                    z += expf(dotProductV[k] - maxDotProduct);
                }
                float regTerm = 0.0f;
                for (int d = 0; d < (rank + 1); d++)
                {
                    regTerm += powf(beta[subLabels[j]][d],2);
                }
                logLikelihood += dotProductV[subLabels[j]] - (maxDotProduct + logf(z)) - ((0.5f * regParam * regTerm)/(float) noOfTrainingExamples);
            }
            fprintf(outputFile,"Train: LogLikelihood: %f\n", logLikelihood);
        }
        
        if (doTestLikelihood) {
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
                float dotProductV[noLabels];
                float maxDotProduct;
                for (int k = 0; k < noLabels; k++)
                {
                    float dotProduct = 0.0f;
                    for (int d = 0; d < (rank + 1); d++)
                    {
                        dotProduct += beta[k][d] * subHeldFeatures[j][d];
                    }
                    dotProductV[k] = dotProduct;
                    if (k==0) maxDotProduct = dotProduct;
                    if (maxDotProduct < dotProduct) {
                        maxDotProduct = dotProduct;
                    }
                }
                for (int k = 0; k < noLabels; k++)
                {
                    z += expf(dotProductV[k] - maxDotProduct);
                }
                heldLogLikelihood += dotProductV[subHeldLabels[j]] - (maxDotProduct + logf(z));
            }
            fprintf(outputFile,"Heldout: LogLikelihood: %f\n", heldLogLikelihood);
        }
        
        //Check Regularization
        if (debugReg)
        {
            float totalBeta = 0.0f;
            for (int k = 0; k < noLabels; k++)
            {
                for (int j = 0; j < rank + 1; j++)
                {
                    totalBeta += beta[k][j];
                }
            }
            fprintf(outputFile,"Total beta: %f", totalBeta);
        }
        
        fflush(outputFile);
    }
    
    free(subLabels);
    free(subFeatures);
    return 0;
}