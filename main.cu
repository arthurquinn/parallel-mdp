#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include "mdp_structs.h"

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
    std::cout << "./mdp -tmodel <transition model file> -reward <reward definition file>\n" << std::endl;
}

int main(int argc, char ** argv) {

    char * arg;

    struct reward * reward_def;
    struct transition * tmodel;

    // Create array of reward definitions from file
    if ((arg = getCmdOption(argv, argv+argc, "-reward"))) {
        int numstates = 0;
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
        std::cout << arg << std::endl;
    } else {
        errorMsg();
        return(EXIT_FAILURE);
    }



    return 0;
}