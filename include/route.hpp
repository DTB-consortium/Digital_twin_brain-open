#pragma once

#include <vector>
#include <stdint.h>

namespace dtb {

void generate_route( const int32_t n, 
				const std::vector<int32_t>& dimensions,
				std::vector<int32_t>& stage_routes);

}
