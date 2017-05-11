#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "mdp_structs.h"

#define MDP_INFINITY 999999999


struct timeval StartingTime;

void setTime(){
	gettimeofday( &StartingTime, NULL );
}

double getTime(){
	struct timeval PausingTime, ElapsedTime;
	gettimeofday( &PausingTime, NULL );
	timersub(&PausingTime, &StartingTime, &ElapsedTime);
	return ElapsedTime.tv_sec*1000.0+ElapsedTime.tv_usec/1000.0;	// Returning in milliseconds.
}

__global__ void mdp_sum_actions(
	const int numtransitions,
	const int numactions,
	const struct transition * tmodel,
	const float * util_prev,
	float * action_utils) {

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;

	int iter = numtransitions % thread_num ? numtransitions / thread_num + 1 : numtransitions / thread_num;

	// Each thread is assigned a transition
	// In each thread we must update the utility values of each action for each state
	for (int i = 0; i < iter; i++) {

		int dataid = thread_id + i * thread_num;
		if (dataid < numtransitions) {

			int action = tmodel[dataid].a;
			int state = tmodel[dataid].s;
			int tstate = tmodel[dataid].sp;
			atomicAdd(&action_utils[state * numactions + action], tmodel[dataid].p * util_prev[tstate]);
		}
	}
}

__global__ void mdp_update_util(
	const int numstates,
	const int numactions,
	const float discount,
	const struct reward * reward_def,
	float * util_curr,
	const float * action_utils) {

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;

	int iter = numstates % thread_num ? numstates / thread_num + 1 : numstates / thread_num;

	// Once action util values are calculated, select the max to update the utility of the state s
	// Here we consider each thread as a state rather than as a transition (above)
	for (int i = 0; i < iter; i++) {
		int state = thread_id + i * thread_num;
		if (state < numstates) {
			float amax = -MDP_INFINITY;
			for (int j = 0; j < numactions; j++) {
				amax = max(amax, action_utils[state * numactions + j]);
			}
			util_curr[state] = reward_def[state].reward + discount * amax;
		}
	}
}

double mdp(
	const int numstates, 
	const int numtransitions,
	const int numactions,
	const float epsilon,
	const float discount,
	const int numBlocks,
	const int blockSize,
	const struct transition * tmodel,
	const struct reward * reward_def,
	float * util_curr) {

	double elapsed_time = 0.0;

	float * util_prev = (float *)malloc(sizeof(float) * numstates);
	memcpy(util_prev, util_curr, sizeof(float) * numstates);

	struct transition * d_tmodel;
	struct reward * d_reward_def;
	float * d_util_prev;
	float * d_util_curr;
	float * d_action_utils;
	cudaMalloc((void **)&d_tmodel, numtransitions * sizeof(struct transition));
	cudaMalloc((void **)&d_reward_def, numstates * sizeof(struct reward));
	cudaMalloc((void **)&d_util_prev, numstates * sizeof(float));
	cudaMalloc((void **)&d_util_curr, numstates * sizeof(float));
	cudaMalloc((void **)&d_action_utils, numstates * numactions * sizeof(float));
	cudaMemcpy(d_tmodel, tmodel, numtransitions * sizeof(struct transition), cudaMemcpyHostToDevice);
	cudaMemcpy(d_reward_def, reward_def, numstates * sizeof(struct reward), cudaMemcpyHostToDevice);
	cudaMemcpy(d_util_prev, util_prev, numstates * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_util_curr, util_curr, numstates * sizeof(float), cudaMemcpyHostToDevice);

	float delta = -1.0;


	int iter = 0;
	do {

		// for (int i = 0; i < numstates; i++) {
		// 	printf("%f\n", util_prev[i]);
		// }

		// for (int i = 0; i < numstates; i++) {
		// 	printf("%f\n", util_curr[i]);
		// }

		//setTime();

		cudaMemset(d_action_utils, 0, numstates * numactions * sizeof(float));

		//setTime();
		mdp_sum_actions<<<numBlocks, blockSize>>>(
			numtransitions, numactions, d_tmodel, d_util_prev, d_action_utils);
		cudaDeviceSynchronize();
		//std::cout << "Took: " << getTime() << " ms." << std::endl;

		setTime();
		mdp_update_util<<<numBlocks, blockSize>>>(
			numstates, numactions, discount, d_reward_def, d_util_curr, d_action_utils);
		//cudaDeviceSynchronize();
		//std::cout << "Took: " << getTime() << " ms." << std::endl;
		//elapsed_time += getTime();
		cudaMemcpy(util_curr, d_util_curr, numstates * sizeof(float), cudaMemcpyDeviceToHost);
		//sleep(1);

		// for (int i = 0; i < numstates; i++) {
		// 	printf("%f\n", util_prev[i]);
		// }

		// for (int i = 0; i < numstates; i++) {
		// 	printf("%f\n", util_curr[i]);
		// }

		delta = -1.0;
		for (int i = 0; i < numstates; i++) {
			delta = max(delta, abs(util_curr[i] - util_prev[i]));
		}

		std::cout << "Delta = " << delta << std::endl;
		std::cin.get();

		if (delta < epsilon * (1.0 - discount) / discount) {
			break;
		}

		float * temp = util_prev;
		util_prev = util_curr;
		util_curr = temp;

		cudaMemcpy(d_util_prev, util_prev, numstates * sizeof(float), cudaMemcpyHostToDevice);
		iter++;
	} while (true);

	std::cout << "" << iter << std::endl;
	return elapsed_time;

}