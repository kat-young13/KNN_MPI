#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include "mpi.h"

using namespace std;

int* KNN(ArffData* dataset, int k_neighbors, int rank, int numtasks)
{
    int* predictions = (int*)calloc(dataset->num_instances(), sizeof(int));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Status Stat;
    int chunk_size = dataset->num_instances() / (numtasks);
    int* predictions_from_rank =(int*)calloc(chunk_size, sizeof(int));


    if(rank >= 0){ 
       for(int i = (rank)*chunk_size; i < (rank+1)*chunk_size && i < dataset->num_instances(); i++){ //for each instance in chunk
	float* Arr_dist =(float*)calloc(k_neighbors, sizeof(float));
	int* Arr_classes=(int*)calloc(k_neighbors, sizeof(int));
        float largest_array_distance = 0;
        int index_largest_distance = 0;

        for(int j = 0; j < dataset->num_instances(); j++){ // look at all other instances
	  if(i==j) continue;
	  float distance = 0;

	  for(int k = 0; k < dataset->num_attributes() - 1; k++){ // compute distance between two instances
	    float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
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

	  if(j >= k_neighbors){
	    if(distance < largest_array_distance){ // IF THERE IS A CLOSER NEIGHBOR THAT SHOULD BE IN ARRAY
	      Arr_dist[index_largest_distance] = distance; // change the distance, then add the class
	      Arr_classes[index_largest_distance] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32();
	      //FIND NEW LARGEST DISTANCE
	      float new_largest = 0;
	      int new_largest_index;
	      for(int r = 0; r < k_neighbors; r++){
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
        }
	
        // VOTE ON THE CLASS!!! array of size k(each slot in array corresponds to class), when class found in array, increment index.
        int* classVotes =(int*)calloc(dataset->num_classes(), sizeof(int));
        for(int g = 0; g < k_neighbors; g++){ // go through each nearest neighbor
	  classVotes[Arr_classes[g]] += 1;
        }

        int max_Votes = 0;
        int max_Votes_index = 0;
        for(int f = 0; f < dataset->num_classes(); f++){
	  if(classVotes[f] > max_Votes){
	    max_Votes = classVotes[f];
	    max_Votes_index = f;
	  }
        }
	predictions_from_rank[i % chunk_size] = max_Votes_index;
	}
    }

    MPI_Gather (predictions_from_rank, chunk_size, MPI_INT, predictions, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // see whats in predictions array being returned
     if(rank == 0) {
       if( chunk_size*numtasks < dataset->num_instances()){
	 int remainder = dataset->num_instances() % (chunk_size * numtasks);
	 float* Arr_dist_rem =(float*)calloc(k_neighbors, sizeof(float));
	 int* Arr_classes_rem =(int*)calloc(k_neighbors, sizeof(int));
	 float largest_array_distance_rem = 0;
	 int index_largest_distance_rem = 0;

	 for(int i = dataset->num_instances() - remainder - 1; i < dataset->num_instances(); i++){	   
	   for(int j = 0; j < dataset->num_instances(); j++){ // look at all other instances                                                          
	     if(i==j) continue;
	     float distance = 0;
	     for(int k = 0; k < dataset->num_attributes() - 1; k++){ // compute distance between two instances                                        
	       float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
	       distance += diff * diff;
	     }
	     distance = sqrt(distance);

	     // THIS IS INITIALIZING ARRAY AUTOMATICALLY WITH FIRST K VALUES. Get a starting point.                                                   
	     if(j < k_neighbors){
	       Arr_dist_rem[j] = distance; //put distance in                                                                                          
	       Arr_classes_rem[j] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32(); // put class in            
	       if(distance > largest_array_distance_rem){ // keep track of largest distance in array                                                     
		 largest_array_distance_rem = distance;
		 index_largest_distance_rem = j;
	       }
	     }

	     if(j >= k_neighbors){
	       if(distance < largest_array_distance_rem){ // IF THERE IS A CLOSER NEIGHBOR THAT SHOULD BE IN ARRAY                   
		 Arr_dist_rem[index_largest_distance_rem] = distance; // change the distance, then add the class                                      
		 Arr_classes_rem[index_largest_distance_rem] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32();
		 //FIND NEW LARGEST DISTANCE                                                                                                          
		 float new_largest_rem = 0;
		 int new_largest_index_rem;
		 for(int r = 0; r < k_neighbors; r++){
		   float temp = Arr_dist_rem[r];
		   if(temp > new_largest_rem){
		     new_largest_rem = temp;
		     new_largest_index_rem = r;
		   }
		 }
	       }

	       int* classVotes_rem =(int*)calloc(dataset->num_classes(), sizeof(int));
	       for(int g = 0; g < k_neighbors; g++){ // go through each nearest neighbor                                                                      
		 classVotes_rem[Arr_classes_rem[g]] += 1;
	       }

	       int max_Votes_rem = 0;
	       int max_Votes_index_rem = 0;
	       for(int f = 0; f < dataset->num_classes(); f++){
		 if(classVotes_rem[f] > max_Votes_rem){
		   max_Votes_rem = classVotes_rem[f];
		   max_Votes_index_rem = f;
		 }
	       }
	       predictions[i] = max_Votes_index_rem;
	     }
	   }



	 }
       }
     return predictions;
   }
    else {
      return NULL;
      }
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

    //int rank, numtasks;
    MPI_Init(&argc,&argv);
    int rank, numtasks;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
   
    int* predictions = KNN(dataset, 5, rank, numtasks);

    if(predictions != NULL) {

      int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
      float accuracy = computeAccuracy(confusionMatrix, dataset);
    
      clock_gettime(CLOCK_MONOTONIC_RAW, &end);
      uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    
    if(rank == 0){
      printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);

      }
    //cout << "rank is ... " << rank << endl;
    //printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);

    }

    MPI_Finalize();

}
