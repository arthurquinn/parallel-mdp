#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include "mdp_structs.h"

#include "mdp.cu"
#include "mdp-sequential.cpp"

char* getCmdOption(char ** begin, char ** end, const std::string & option) {

    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char ** begin, char ** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

void errorMsg() {
    std::cout << "Missing required command line arguments" << std::endl;
    std::cout << "Usage must be of the following format:\n" << std::endl;
    std::cout << "./mdp -tmodel <transition model file> -reward <reward definition file> -output <output file name> -blockSize <block size> -blockNum <block num>\n" << std::endl;
}

int main(int argc, char ** argv) {

    char * arg;

    char * outfile;

    int blockNum = 1024;
    int blockSize = 1024;

    struct reward * reward_def;
    struct transition * tmodel;
    int numstates = 0;
    int numtransitions = 0;
    int numactions = 0;

    // Create array of reward definitions from file
    if ((arg = getCmdOption(argv, argv+argc, "-reward"))) {
        std::string temp;
        std::ifstream infile;
        infile.open(std::string(arg));

        int i = 0;
        while (!infile.eof()) {
            std::getline(infile,temp);
            // First line is number of states
            if (i == 0) {
                numstates = std::stoi(temp);
                reward_def = (struct reward *)malloc(sizeof(struct reward) * numstates);
            } else if (!temp.empty()) {
                char * cstr = (char *)temp.c_str();
                char * tok = strtok(cstr, " ");
                reward_def[i - 1].s = atoi(tok);

                tok = strtok(NULL, " ");
                reward_def[i - 1].reward = atof(tok);
            }
            i++;
        }
        infile.close();
        std::cout << "Created reward array with " << numstates << " reward definitions." << std::endl;
    } else {
        errorMsg();
        return(EXIT_FAILURE);
    }

    // Create an array to hold the transition model from file
    if ((arg = getCmdOption(argv, argv+argc, "-tmodel"))) {

        std::string temp;
        std::ifstream infile;
        infile.open(std::string(arg));

        int i = 0;
        while (!infile.eof()) {
            std::getline(infile, temp);

            // First line is number of transitions
            if (i == 0) {
                char * cstr = (char *)temp.c_str();
                char * tok = strtok(cstr, " ");
                numtransitions = atoi(tok);

                tok = strtok(NULL, " ");
                numactions = atoi(tok);

                tmodel = (struct transition *)malloc(sizeof(struct transition) * numtransitions);
            } else if (!temp.empty()) {
                char * cstr = (char *)temp.c_str();
                char * tok = strtok(cstr, " ");
                tmodel[i - 1].s = atoi(tok);

                tok = strtok(NULL, " ");
                tmodel[i - 1].a = atoi(tok);

                tok = strtok(NULL, " ");
                tmodel[i - 1].sp = atoi(tok);

                tok = strtok(NULL, " ");
                tmodel[i - 1].p = atof(tok);
            }
            i++;
        }

        std::cout << "Created transition model with " << numtransitions << " transitions and " << numactions << " actions." << std::endl;
    } else {
        errorMsg();
        return(EXIT_FAILURE);
    }

    // Prepare output file
    if ((arg = getCmdOption(argv, argv+argc, "-output"))) {
        outfile = arg;
    } else {
        errorMsg();
        return(EXIT_FAILURE);
    }

    // Get block size and block num
    if ((arg = getCmdOption(argv, argv+argc, "-blockSize"))) {
        blockSize = atoi(arg);
    } else {
        errorMsg();
        return(EXIT_FAILURE);
    }

    if ((arg = getCmdOption(argv, argv+argc, "-blockNum"))) {
        blockNum = atoi(arg);
    } else {
        errorMsg();
        return(EXIT_FAILURE);
    }

    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Block Num: " << blockNum << std::endl;

    const float epsilon = 0.001;
    const float discount = 0.8;

    // Instantiate array where final utilities will be stored
    float * final_utilities = (float *)calloc(numstates, sizeof(float));
    float * actual_utilities = (float *)calloc(numstates, sizeof(float));

    double kernel_time = mdp(numstates, numtransitions, numactions, epsilon, discount, blockNum, blockSize, tmodel, reward_def, final_utilities);

    double seq_time = mdp_seq(numstates, numtransitions, numactions, epsilon, discount, tmodel, reward_def, actual_utilities);

    std::ofstream outputFile;
    outputFile.open(outfile);
    for (int i = 0; i < numstates; i++) {
        outputFile << "" << i << " " << final_utilities[i] << std::endl;
    }
    outputFile.close();

    // Get number of states that have erroneous utilities
    float error_bound = 0.01;
    int numerr = 0;
    for (int i = 0; i < numstates; i++) {
        if (abs(final_utilities[i] - actual_utilities[i]) > error_bound) {
            numerr++;
        }
    }

    std::cout << "Total kernel time: " << kernel_time << "ms" << std::endl;
    std::cout << "Sequential time: " << seq_time << "ms" << std::endl;
    std::cout << "Percentage of utilities U(s) calculated incorrectly (0.01 tolerance): " << (float)numerr / (float)numstates << "%" << std::endl;

    return(EXIT_SUCCESS);
}