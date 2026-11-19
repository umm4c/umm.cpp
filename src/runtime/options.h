#pragma once

#include <map>
#include <string>

namespace umm::runtime {

using options = std::map<std::string, std::string>;

options parse_options(int argc, char ** argv);
std::string option(const options &, const std::string &, const std::string & fallback = "");

}
