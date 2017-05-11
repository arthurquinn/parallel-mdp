CC = /usr/local/cuda-7.5/bin/nvcc

mdp: *.cu *.cpp *.h
	$(CC) -std=c++11 main.cu -O3 -arch=sm_30 -o mdp

clean:
	rm -f *.o mdp

test:
	./mdp -tmodel small-transition.txt -reward small-reward.txt