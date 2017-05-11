CC = /usr/local/cuda-7.5/bin/nvcc

mdp: *.cu *.h
	$(CC) -std=c++11 main.cu -O3 -arch=sm_30 -o mdp

clean:
	rm -f *.o mdp

test:
	./mdp -tmodel sample1-transition.txt -reward sample1-reward.txt