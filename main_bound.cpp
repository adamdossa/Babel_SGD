//
//  main.cpp
//  Babel_SGD
//
//  Created by Adam Dossa on 11/06/2013.
//  Copyright (c) 2013 Adam Dossa. All rights reserved.
//
//  Implementation of stocastic bound method (as per NIPS2012 Tony Jebara / Anna C Paper)

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
//#include <libc.h>
#include <netinet/in.h>

#define FEATURE_COLS 360
#define RANK 360
#define NUM_FILES 1
#define NUM_HELDOUT_FILES 1
//#define NUM_FILES 30700
//#define NUM_HELDOUT_FILES 4190
//#define NUM_CLASSES 1000
#define NUM_CLASSES 2
#define NUM_EPOCHS 30
#define INITIAL_LR 0.001f
#define REG_PARAM 100.0f

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
    bool debugLoadings = false;
    bool timeLoops = false;
    bool debugBound = true;
    bool doPCA = false;
    const char * heldFileName = "filterh";
    const char * trainFileName = "filter";
    
    //Defaults (for my mac)
    const char * rootDirectory = "/Users/AdamDossa/Documents/Columbia/GRA/Babel/";
    const char * currentDirectory = "/Users/AdamDossa/Documents/XCode/Babel_SGD/Babel_SGD/log/";
    float regParam = REG_PARAM;
    float initialLR = INITIAL_LR;
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
        regParam = (float) atof(argv[3]);
    }
    if (argc > 4)
    {
        initialLR = (float) atof(argv[4]);
    }
    if (argc > 5)
    {
        rank = (int) atoi(argv[5]);
    }
    
    //We want to find an unused log file name (to some limit)
    char logFileName[200];
    for (int i = 0; i < 100; i++)
    {
        snprintf(logFileName, sizeof(char) * 200,"%s/log/log_bound-%d-%d-%d-%f-%f.txt.%d", currentDirectory, rank, NUM_EPOCHS, NUM_FILES, regParam, initialLR, i);
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
    
    //First read in the loadings, must have been generated for correct number of input files (dubious)
    fprintf(outputFile, "Reading in loadings\n");
    double ** loadings = (double **) malloc(sizeof(double *) * FEATURE_COLS);
    if (doPCA) {
        char loadName[200];
        snprintf(loadName, sizeof(char) * 200,"%s/loadsings/loadings-%d.load", currentDirectory, NUM_FILES);
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
    }
    fflush(outputFile);
        
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
    fprintf(outputFile, "No of total training examples: %d\n", noOfTrainingExamples);
    fflush(outputFile);
        
    //First read in the labels
    fprintf(outputFile, "Reading in labels\n");
    int readSoFar = 0;
    int * labels = (int *) malloc(sizeof(int) * noOfTrainingExamples);
    for (int i = 0; i < NUM_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/labels/%s.%d.lab", rootDirectory, trainFileName, i);
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
    //We apply loadings as we store each sample
    fprintf(outputFile,"Reading in features\n");
    readSoFar = 0;
    float ** features = (float **) malloc(sizeof(float *) * noOfTrainingExamples);
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
            features[readSoFar + j] = (float *) malloc(sizeof(float) * (rank + 1));
            for (int k = 0; k < rank; k++)
            {
                float featureI = 0.0f;
                if (doPCA) {
                    for (int p = 0; p < m; p++)
                    {
                        featureI += ((float) loadings[p][k]) * bin2flt(&tempFeatures[p]);
                    }
                } else {
                    featureI = bin2flt(&tempFeatures[k]);
                }
                features[readSoFar + j][k] = featureI;
            }
            free(tempFeatures);
            features[readSoFar + j][rank] = 1.0f;
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
    
    //Now do the same for the heldout data
    //First calculate the number of training examples (since not every file is guaranteed to have 250 rows)
    fprintf(outputFile, "Calculating number of heldout examples\n");
    int noOfHeldoutExamples = 0;
    for (int i = 0; i < NUM_HELDOUT_FILES; i++) {
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/labels/%s.%d.lab", rootDirectory, heldFileName, i);
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
        snprintf(fName, sizeof(char) * 200,"%s/labels/%s.%d.lab", rootDirectory, heldFileName, i);
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
        snprintf(fName, sizeof(char) * 200,"%s/features/%s.%d.fea", rootDirectory, heldFileName, i);
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
            heldFeatures[readSoFar + j] = (float *) malloc(sizeof(float) * (rank + 1));
            for (int k = 0; k < rank; k++)
            {
                float featureI = 0.0f;
                if (doPCA) {
                    for (int p = 0; p < m; p++)
                    {
                        featureI += ((float) loadings[p][k]) * bin2flt(&tempFeatures[p]);
                    }
                } else {
                    featureI = bin2flt(&tempFeatures[k]);
                }
                heldFeatures[readSoFar + j][k] = featureI;
            }
            heldFeatures[readSoFar + j][rank] = 1.0f;
            free(tempFeatures);
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
    
    //Now run Stochastic Bound Method
    fprintf(outputFile,"Running Stochastic Bound Method\n");
    fflush(outputFile);
    
    //Initialize theta parameters, one for each class
    float ** theta = (float **) malloc(sizeof(float *) * NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        theta[i] = (float *) malloc(sizeof(float) * (rank + 1));
        for (int j = 0; j < (rank + 1); j++)
        {
            theta[i][j] = 0.0f;
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
    //Check for any theta* files of appropriate name, and initialize using latest epoch
    int startingEpoch = 0;
    char lsName[200];
    snprintf(lsName, sizeof(char) * 200,"ls %s/beta/theta_bound-%d-%s-%d-%f-%f.out | grep 'theta_bound' | cut -d'-' -f3", currentDirectory, rank, "*", NUM_FILES, regParam, initialLR);
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
    
    //Now read in theta files as needed
    if (startingEpoch > 0)
    {
        //Initialize using theta file
        snprintf(lsName, sizeof(char) * 200,"%s/beta/theta_bound-%d-%d-%d-%f-%f.out", currentDirectory, rank, startingEpoch - 1, NUM_FILES, regParam, initialLR);
        fprintf(outputFile,"Initializing from file: %s\n", lsName);
        lsFile = fopen(lsName,"rb");
        for (int k = 0; k < NUM_CLASSES; k++)
        {
            fread(theta[k],sizeof(float),rank + 1, lsFile);
        }
        
    }
    fflush(outputFile);
    
    //Initialize remaining outerloop variables as per paper
    int dim = (rank + 1) * NUM_CLASSES;
    float * phi = (float *) malloc(sizeof(float) * dim);
    for (int j = 0; j < dim; j++)
    {
        phi[j] = 0.0f;
    }

    float ** M = (float **) malloc(sizeof(float *) * dim);
    for (int i = 0; i < dim; i++)
    {
        M[i] = (float *) malloc(sizeof(float) * dim);
        for (int j = 0; j < dim; j++)
        {
            if (i==j) {
                M[i][j] = 1.0f / regParam;
            } else {
                M[i][j] = 0.0f;
            }
        }
    }
    
    float * mu = (float *) malloc(sizeof(float) * dim);
    for (int j = 0; j < dim; j++)
    {
        mu[j] = 0.0f;
    }
    
    float * g = (float *) malloc(sizeof(float) * dim);
    float * l = (float *) malloc(sizeof(float) * dim);
    float * xi = (float *) malloc(sizeof(float) * dim);
    float * ltm = (float *) malloc(sizeof(float) * dim);
    float ** N = (float **) malloc(sizeof(float *) * dim);
    for (int i = 0; i < dim; i++)
    {
        N[i] = (float *) malloc(sizeof(float) * dim);
    };
    //Start looping for NUM_EPOCHS
    for (int e = startingEpoch; e < NUM_EPOCHS; e++)
    {
        fprintf(outputFile,"Bound Epoch: %d\n", e);
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
            
            //Initialize inner variables as per paper
            float z = -80.0f; //log of a small number close to 0
            
            //Zero out g variable
            for (int d = 0; d < dim; d++)
            {
                g[d] = 0.0f;
            }
            
            for (int k = 0; k < NUM_CLASSES; k++)
            {
                if (debugBound) {
                    fprintf(outputFile, "Running epoch: %d, sample: %d, class: %d\n", e, j, k);
                    fflush(outputFile);
                }
                float alpha = 0.0f;
                for (int d = 0; d < (rank + 1); d++)
                {
                    alpha += theta[k][d] * features[j][d];
                }

                for (int d = 0; d < NUM_CLASSES; d++)
                {
                    for (int p = 0; p < (rank + 1); p++)
                    {
                        if (d == k) {
                            l[(d * (rank + 1)) + p] = features[j][p] - g[(d * (rank + 1)) + p];
                        } else {
                            l[(d * (rank + 1)) + p] = 0.0f - g[(d * (rank + 1)) + p];
                        }
                    }
                }
                
                float beta;
                if (alpha == z) {
                    beta = 0.25f;
                } else {
                    beta = tanhf(0.5f * (alpha - z))/(2 * (alpha - z));
                }
                float kappa;
                if (z > alpha) {
                    float rat1 = 1.0f/(1.0f+expf(alpha - z));
//                    float rat2 = 1.0f - rat1;
                    kappa = (1 - rat1);
                } else {
                    float rat2 = 1.0f/(1.0f+expf(z - alpha));
                    float rat1 = 1.0f - rat2;
                    kappa = (1 - rat1);
                }
                
                for (int d = 0; d < NUM_CLASSES; d++)
                {
                    for (int p = 0; p < (rank + 1); p++)
                    {
                        if (d == labels[j]) {
                            xi[(d * (rank + 1)) + p] = (kappa * l[(d * (rank + 1)) + p]) - (features[j][p]/NUM_CLASSES) + ((regParam * theta[d][p])/noOfTrainingExamples);
                        } else {
                            xi[(d * (rank + 1)) + p] = (kappa * l[(d * (rank + 1)) + p]) - 0.0f + ((regParam * theta[d][p])/noOfTrainingExamples);
                            
                        }
                    }
                }

                for (int i = 0; i < dim; i++)
                {
                    float ltm_i = 0.0f;
                    for (int p = 0; p < dim; p++)
                    {
                        ltm_i += l[p] * M[p][i];
                    }
                    ltm[i] = beta * ltm_i;
                }
                float denom = 0.0f;
                for (int i = 0; i < dim; i++)
                {
                    denom += ltm[i] * l[i];
                }
                denom += 1;

                for (int i = 0; i < dim; i++)
                {
                    for (int p = 0; p < dim; p++)
                    {
                        float N_i_p = 0.0f;
                        for (int d = 0; d < dim; d++)
                        {
                            N_i_p += M[i][d] * l[d] * ltm[p];
                        }
                        N[i][p] = N_i_p / denom;
                    }
                }
                for (int i = 0; i < dim; i++)
                {
                    for (int p = 0; p < dim; p++)
                    {
                        M[i][p] = M[i][p] - N[i][p];
                    }
                }
                for (int i = 0; i < dim; i++)
                {
                    float phi_i = 0.0f;
                    for (int p = 0; p < dim; p++)
                    {
                        phi_i += (M[i][p] * xi[p]) - (N[i][p] * mu[p]);
                    }
                    phi[i] = phi[i] + phi_i;
                }
                
                for (int i = 0; i < dim; i++)
                {
                    g[i] = g[i] + (kappa * l[i]);
                }
                
                for (int i = 0; i < dim; i++)
                {
                    mu[i] = mu[i] + xi[i];
                }
                
                if (alpha < z) {
                    z = z + logf(1.0f + expf(alpha - z));
                } else {
                    z = alpha + logf(1.0f+expf(z - alpha));
                }
            }
            int theta_row = 0;
            int theta_col = 0;
            for (int d = 0; d < dim; d++)
            {
                if (theta_col == (rank + 1))
                {
                    theta_col = 0;
                    theta_row += 1;
                }
                theta_col += 1;
            }
        }
        if (debugBound) {
            int theta_row = 0;
            int theta_col = 0;

            for (int d = 0; d < dim; d++)
            {
                if (theta_col == (rank + 1))
                {
                    theta_col = 0;
                    theta_row += 1;
                }
                fprintf(outputFile, "Theta: %d, %d: %f\n",theta_row, theta_col,theta[theta_row][theta_col]);
                theta_col += 1;
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
                for (int d = 0; d < (rank + 1); d++)
                {
                    dotProduct += theta[k][d] * features[j][d];
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
            for (int d = 0; d < (rank + 1); d++)
            {
                regTerm += powf(theta[labels[j]][d],2);
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
                for (int d = 0; d < (rank + 1); d++)
                {
                    dotProduct += theta[k][d] * heldFeatures[j][d];
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
            float totaltheta = 0.0f;
            for (int k = 0; k < NUM_CLASSES; k++)
            {
                for (int j = 0; j < rank + 1; j++)
                {
                    totaltheta += theta[k][j];
                }
            }
            fprintf(outputFile,"Total theta: %f", totaltheta);
        }
        
        //Now write to a file named for the epoch etc.
        char fName[200];
        snprintf(fName, sizeof(char) * 200,"%s/beta/theta_bound-%d-%d-%d-%f-%f.out", currentDirectory, rank, e, NUM_FILES, regParam, initialLR);
        pFile = fopen(fName,"wb");
        for (int k = 0; k < NUM_CLASSES; k++)
        {
            fwrite(theta[k],sizeof(float),rank + 1,pFile);
        }
        fclose(pFile);
        fflush(outputFile);
    }
    free(g);
    free(l);
    free(xi);
    free(ltm);
    free(N);
    free(M);
    free(phi);
    free(mu);
    free(theta);
    free(labels);
    free(features);
    free(heldLabels);
    free(heldFeatures);
    return 0;
}