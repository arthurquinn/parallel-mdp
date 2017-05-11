CC = /usr/local/cuda-7.5/bin/nvcc

mdp: *.cu *.cpp 
	$(CC) -std=c++11 main.cu -O3 -arch=sm_30 -o mdp

clean:
	rm -f *.o mdp

test:
	./mdp -tmodel sample4-transition.txt -reward sample4-reward.txt -output sample4-output.txt -blockSize 1024 -blockNum 2