#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;

int* KNN(ArffData* dataset, int k_neighbors)
{
    //THIS LINE: allocates memory for the number of instances in the dataset considering size of instances
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

    for(int i = 0; i < dataset->num_instances(); i++){ //for each instance in dataset
        //float smallestDistance = FLT_MAX;
        //int smallestDistanceClass;
	float Arr_dist[k_neighbors];
        int Arr_classes[k_neighbors];
        float largest_array_distance = 0;
        int index_largest_distance;

        for(int j = 0; j < dataset->num_instances(); j++){ // look at all other instances

            if(i==j) continue;
            float distance = 0;

            for(int k = 0; k < dataset->num_attributes() - 1; k++){ // compute distance between two instances
                float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
                // this gets i, gets feature k, looks at distance from other instance j to k
                distance += diff * diff;
            }
            distance = sqrt(distance);

	    // THIS IS INITIALIZING ARRAY AUTOMATICALLY WITH FIRST K VALUES. Get a starting point.
	    if(j < k_neighbors){ 
                Arr_dist[j] = distance; //put distance in
                Arr_classes[j] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32(); // put class in
		if(distance > largest_array_distance){ // keep track of largest distance in array
		    largest_array_distance = distance;
		    index_largest_distance = j;
		}
            }

	    if(j > k_neighbors){
	        if(distance < largest_array_distance){ // IF THERE IS A CLOSER NEIGHBOR THAT SHOULD BE IN ARRAY
		    Arr_dist[index_largest_distance] = distance; // change the distance, then add the class
		    Arr_classes[index_largest_distance] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32();
		    //FIND NEW LARGEST DISTANCE
		    float new_largest = 0;
		    int new_largest_index;
		    for(int r = 0; r < sizeof(Arr_dist); r++){
		        float temp = Arr_dist[r];
			if(temp > new_largest){
			    new_largest = temp;
			    new_largest_index = r;
			}
 		    }
		    largest_array_distance = new_largest;
		    index_largest_distance  = new_largest_index;
		}
	    }
	    cout << Arr_dist[0] << endl;
        }
    }

    //THIS LINE: how to get instance value for a feature
    //float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();

    //THIS LINE: similar to previous. gets the class. casted as integer.
    //int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    
    // Implement KNN here, fill array of class predictions

    
    return predictions;
}



int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        //cout << trueClass;
        int predictedClass = predictions[i];
        //cout << predictedClass;
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }
    
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
   
    int* predictions = KNN(dataset, 5);
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
    printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}
