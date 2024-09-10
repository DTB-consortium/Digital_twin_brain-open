#include "weights.hpp"
#include "logging.hpp"
#include <climits>

namespace dtb {

size_t data_size(DataType dtype) {
  switch (dtype) {
    case DOUBLE:
      return sizeof(double);
   case FLOAT:
      return sizeof(float);
    case FLOAT16:
      return sizeof(half);
	case INT8:
	  return sizeof(char);
    default:
      LOG_ERROR << "Unsupported math type";
      break;
  }
  return 4;
}

}

