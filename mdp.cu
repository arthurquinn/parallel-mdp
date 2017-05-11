#include <stdlib.h>
#include <string.h>

#include "mdp_structs.h"

#define MDP_INFINITY 999999999

__global__ void mdp_kernel(
	const int numstates,
	const int numtransitions,
	const int numactions,
	const struct transition * tmodel,
	const struct reward * reward_def,
	const int * util_prev,
	int * util_curr) {

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;

	int iter = numtransitions % thread_num ? numtransitions / thread_num + 1 : numtransitions / thread_num;

	// Allocate shared mem for action utility values for each state
	extern __shared__ int action_utils[];	

	// Each thread is assigned a transition
	// In each thread we must update the utility values of each action for each state
	for (int i = 0; i < iter; i++) {

		int dataid = thread_id + i * thread_num;
		if (dataid < numtransitions) {

			int action = tmodel[dataid].a;
			int state = tmodel[dataid].s;
			int tstate = tmodel[dataid].sp;

			action_utils[state + action] = tmodel[dataid].p * util_prev[tstate];	
		}
	}

	// Once action util values are calculated, select the max to update the utility of the state s
	// Here we consider each thread as a state rather than as a transition (above)
	__syncthreads();
	


}

void mdp(
	const int numstates, 
	const int numtransitions,
	const int numactions,
	const int epsilon,
	const int numBlocks,
	const int blockSize,
	const struct transition * tmodel,
	const struct reward * reward_def,
	int * utilities) {

	struct transition * d_tmodel;
	struct reward * d_reward_def;
	int * d_util_prev;
	int * d_util_curr;
	cudaMalloc((void **)&d_tmodel, numtransitions * sizeof(struct transition));
	cudaMalloc((void **)&d_reward_def, numstates * sizeof(struct reward));
	cudaMalloc((void **)&d_util_prev, numstates * sizeof(int));
	cudaMalloc((void **)&d_util_curr, numstates * sizeof(int));
	cudaMemcpy(d_tmodel, tmodel, numtransitions * sizeof(struct transition), cudaMemcpyHostToDevice);
	cudaMemcpy(d_reward_def, reward_def, numstates * sizeof(struct reward), cudaMemcpyHostToDevice);
	cudaMemcpy(d_util_prev, utilities, numstates * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_util_curr, utilities, numstates * sizeof(int), cudaMemcpyHostToDevice);
	
	float delta = MDP_INFINITY;

	do {

		mdp_kernel<<<numBlocks, blockSize, numstates + sizeof(int) * numactions>>>(
			numstates, numtransitions, numactions, d_tmodel, d_reward_def, d_util_prev, d_util_curr);
		//cudaDeviceSynchronize();

	} while (delta > epsilon);

}