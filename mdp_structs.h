#ifndef MDP_STRUCTS
#define MDP_STRUCTS

struct transition {
	unsigned int s;
	unsigned int a;
	unsigned int sp;
	float p;
};

struct reward {
	unsigned int s;
	float reward;
};

#endif