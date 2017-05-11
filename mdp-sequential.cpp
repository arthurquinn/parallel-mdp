#include <stdlib.h>
#include <string.h>

#include "mdp_structs.h"

void mdp_seq(
	const int numstates,
	const int numtransitions,
	const int numactions,
	const float epsilon,
	const float discount,
	const struct transition * tmodel,
	const struct reward * reward_def,
	float * util_curr) {

	float * util_prev = (float *)malloc(numstates * sizeof(float));
	memcpy(util_prev, util_curr, numstates * sizeof(float));

	float * action_utils = (float *)malloc(numactions * numstates * sizeof(float));

	do {
		// Reset action_utils array at each iteration
		memset(action_utils, 0, numstates * numactions * sizeof(float));

		// Use transition model to update utility of each action for each state
		for (int i = 0; i < numtransitions; i++) {
			// Compute new action_utils
			int state = tmodel[i].s;
			int action = tmodel[i].a;

			action_utils[state*numactions + action] += tmodel[i].p * util_prev[tmodel[i].sp];
		}

		// Update the utility of each state by selecting the action with max utility and using update formula
		for (int i = 0; i < numstates; i++) {
			float amax = 0.0;
			for (int j = 0; j < numactions; j++) {
				amax = max(amax, action_utils[i * numactions + j]);
			}
			util_curr[i] = reward_def[i].reward + discount * amax;
		}

		// Check if convergence criterion is met
		// If not, copy U' to U and iterate again
		float delta = -1.0;
		for (int i = 0; i < numstates; i++) {
			delta = max(delta, abs(util_curr[i] - util_prev[i]));
		}

		if (delta < epsilon * (1.0 - discount) / discount) {
			break;
		}

		memcpy(util_prev, util_curr, numstates * sizeof(float));
	} while(true);
}