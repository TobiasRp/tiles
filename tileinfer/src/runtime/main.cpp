#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "tile_runtime (project scaffold)\n"
                     "Planned modes: --describe-target-json, --run-plan, --benchmark\n";
        return 0;
    }
    std::cerr << "tile_runtime execution is not implemented yet.\n";
    return 2;
}
