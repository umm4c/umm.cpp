#pragma once

#include "case-protocol.h"

namespace umm::runtime {

json run_jsonl_case(session &, const json & request, const options &);

}
