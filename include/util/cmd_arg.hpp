#pragma once

namespace dtb {

int get_cmdline_argint(const int argc, const char **argv, const char *str_ref);

float get_cmdline_argfloat(const int argc, const char **argv, const char *str_ref);

bool get_cmdline_argstring(const int argc, const char **argv,
                                     const char *str_ref, char **str_retval);

}//namespace istbi 
