#include <cassert>
#include <sstream>
#include "route.hpp"
#include "logging.hpp"

namespace dtb {

void generate_route( const int32_t n, 
				const std::vector<int32_t>& dimensions,
				std::vector<int32_t>& stage_routes)
{
	int dims = dimensions.size();
	std::vector<std::vector<int32_t>> rank_coords;
	rank_coords.resize(n);

	auto transform_coordinate = [dims, &dimensions](int32_t rank, std::vector<int32_t>& coord){
		for(int idx = dims - 1; idx >= 0; idx--)
		{
			coord[idx] = rank % dimensions[idx];
			rank /= dimensions[idx];
		}	
	};

	auto get_rank_num = [&dimensions](std::vector<int32_t>& rank_coords)-> int32_t {
		int32_t rank = 0;
		int32_t count = 1;
		for(int idx = static_cast<int>(rank_coords.size()) - 1; idx >= 0; idx--)
		{
			rank += (rank_coords[idx] * count);
			count *= dimensions[idx];
		}

		return rank;
	};

	for(int32_t idx = 0; idx < n; idx++)
	{
		rank_coords[idx].resize(dims);
		transform_coordinate(idx, rank_coords[idx]);
	}

	std::stringstream ss;
	ss << "coordinate: " << std::endl;
	for(int i = 0; i < rank_coords.size(); i++)
	{
		ss << "(" << rank_coords[i][0] << ", " << rank_coords[i][1] << ")" << " ";
	}
	ss << std::endl;
	LOG_INFO << ss.str();

	stage_routes.resize((n * (n - 1)));

	for(int32_t i = 0; i < n; i++)
	{
		for(int32_t j = 0; j < (n - 1); j++)
		{
			std::vector<int32_t> rank_coord_i = rank_coords[i];
			std::vector<int32_t> rank_coord_j;
			int out_rank = (j >= i) ? (j + 1) : j;
			assert(out_rank < n);
			rank_coord_j = rank_coords[out_rank];
			

			for(size_t k = 0; k < rank_coord_j.size(); k++)
			{
				if(rank_coord_i[k] == rank_coord_j[k])
				{
					continue;
				}

				rank_coord_i[k] = rank_coord_j[k];
				int32_t rank = get_rank_num(rank_coord_i);
				
				if(rank == out_rank)
				{
					stage_routes[i * (n - 1) + j] = i; 
				}
				else
				{
					stage_routes[i * (n - 1) + j] = rank;
				}
				break;
			}
		}
	}
}

}
