# CMake generated Testfile for 
# Source directory: /home/dan/workspace/tu_exs/PPexs/PP1/code/tests/compilation
# Build directory: /home/dan/workspace/tu_exs/PPexs/PP1/code/build/tests/compilation
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_compilation "/usr/bin/cmake" "-DBUILD_DIR=/home/dan/workspace/tu_exs/PPexs/PP1/code/build" "-DTARGET=test_compilation" "-DCONFIG=" "-P" "/home/dan/workspace/tu_exs/PPexs/PP1/code/tests/cmake/BuildAndRun.cmake")
set_tests_properties(test_compilation PROPERTIES  LABELS "interface" RESOURCE_LOCK "dnn_build" _BACKTRACE_TRIPLES "/home/dan/workspace/tu_exs/PPexs/PP1/code/tests/compilation/CMakeLists.txt;38;add_test;/home/dan/workspace/tu_exs/PPexs/PP1/code/tests/compilation/CMakeLists.txt;0;")
