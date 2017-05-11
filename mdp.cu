#include <stdlib.h>
#include <string.h>

#include "mdp_structs.h"

#define MDP_INFINITY 999999999

__global__ void mdp_kernel(
	const int numstates,
	const int numtransitions,
	const int numactions,
	const float discount,
	const struct transition * tmodel,
	const struct reward * reward_def,
	const float * util_prev,
	float * util_curr) {

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;

	int iter = numtransitions % thread_num ? numtransitions / thread_num + 1 : numtransitions / thread_num;

	// Allocate shared mem for action utility values for each state
	extern __shared__ float action_utils[];	

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

	// Once action util values are calculated, select the max to update the utility of the state s
	// Here we consider each thread as a state rather than as a transition (above)
	__syncthreads();
	iter = numstates % thread_num ? numstates / thread_num + 1 : numstates / thread_num;
	for (int i = 0; i < iter; i++) {
		int state = thread_id + i * thread_num;
		if (state < numstates) {
			float amax = 0.0;
			for (int j = 0; j < numactions; j++) {
				amax = max(amax, action_utils[state * numactions + j]);
			}
			util_curr[state] = reward_def[state].reward + 0.5 * amax;
		}
	}

}

void mdp(
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

	float * util_prev = (float *)malloc(sizeof(float) * numstates);
	memcpy(util_prev, util_curr, sizeof(float) * numstates);

	struct transition * d_tmodel;
	struct reward * d_reward_def;
	float * d_util_prev;
	float * d_util_curr;
	cudaMalloc((void **)&d_tmodel, numtransitions * sizeof(struct transition));
	cudaMalloc((void **)&d_reward_def, numstates * sizeof(struct reward));
	cudaMalloc((void **)&d_util_prev, numstates * sizeof(float));
	cudaMalloc((void **)&d_util_curr, numstates * sizeof(float));
	cudaMemcpy(d_tmodel, tmodel, numtransitions * sizeof(struct transition), cudaMemcpyHostToDevice);
	cudaMemcpy(d_reward_def, reward_def, numstates * sizeof(struct reward), cudaMemcpyHostToDevice);
	cudaMemcpy(d_util_prev, util_prev, numstates * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_util_curr, util_curr, numstates * sizeof(float), cudaMemcpyHostToDevice);
	
	float delta = -1.0;

	do {

		for (int i = 0; i < numstates; i++) {
			printf("%f\n", util_prev[i]);
		}

		for (int i = 0; i < numstates; i++) {
			printf("%f\n", util_curr[i]);
		}

		mdp_kernel<<<numBlocks, blockSize, numstates + sizeof(float) * numactions>>>(
			numstates, numtransitions, numactions, discount, d_tmodel, d_reward_def, d_util_prev, d_util_curr);
		cudaDeviceSynchronize();


		cudaMemcpy(util_curr, d_util_curr, numstates * sizeof(float), cudaMemcpyDeviceToHost);

		for (int i = 0; i < numstates; i++) {
			printf("%f\n", util_prev[i]);
		}

		for (int i = 0; i < numstates; i++) {
			printf("%f\n", util_curr[i]);
		}

		delta = -1.0;
		for (int i = 0; i < numstates; i++) {
			delta = max(delta, abs(util_curr[i] - util_prev[i]));
		}

		std::cout << "Delta = " << delta << std::endl;
		std::cin.get();

		std::cout << "cmp: " << (epsilon * (1.0 - discount) / discount) << std::endl;

		if (delta < epsilon * (1.0 - discount) / discount) {
			break;
		}

		float * temp = util_prev;
		util_prev = util_curr;
		util_curr = temp;

		cudaMemcpy(d_util_prev, util_prev, numstates * sizeof(float), cudaMemcpyHostToDevice);

	} while (true);

}