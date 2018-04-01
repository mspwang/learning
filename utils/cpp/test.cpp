#include "cpp_util.h"
#include <iostream>
using namespace pengwa::utils;

int main(int argc, char** argv) {
  Util_Cpp util;
  std::cout << "print out the generated string:" <<
    util.GenerateStringWithTicks("prefixtest", "postfixtest");
  return 0;
}
