#include "cpp_util.h"
#include <chrono>

using namespace std::chrono;
namespace pengwa {
namespace utils {

std::string Util_Cpp::GenerateStringWithTicks(std::string prefix, std::string postfix){
  milliseconds ms = duration_cast< milliseconds >(
    system_clock::now().time_since_epoch()
  );

  std::string new_str = prefix + std::to_string(ms.count()) + postfix;
  return new_str;
}
}
}
