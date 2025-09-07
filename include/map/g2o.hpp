#pragma once
#include <opencv2/core.hpp>
#include "map/map.hpp"  

namespace vo_core {

bool run_g2o_local_BA(BAInput& in, int max_iters = 10);

} // namespace vo_core
