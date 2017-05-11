CC = /usr/local/cuda-7.5/bin/nvcc

mdp: *.cu *.cpp 
	$(CC) -std=c++11 main.cu -O3 -arch=sm_30 -o mdp

clean:
	rm -f *.o mdp

test:
	./mdp -tmodel sample2-transition.txt -reward sample2-reward.txt -output sample2-output.txt