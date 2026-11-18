#include "cli.h"

#include <exception>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        return umm::cli::run(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << "umm: " << error.what() << '\n';
        return 1;
    }
}
