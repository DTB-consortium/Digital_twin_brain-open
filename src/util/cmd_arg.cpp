#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/cmd_arg.hpp"

namespace dtb {

static int remove_delimiter_from_string(char delimiter, const char *str)
{
    int start = 0;

    while (str[start] == delimiter)
    {
        start++;
    }

    if (start >= (int)strlen(str)-1)
    {
        return 0;
    }

    return start;
}

int get_cmdline_argint(const int argc, const char **argv, const char *str_ref)
{
    bool found = false;
    int value = -1;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int start = remove_delimiter_from_string('-', argv[i]);
            const char *str_argv = &argv[i][start];
            int length = (int)strlen(str_ref);

            if (!strncasecmp(str_argv, str_ref, length))
            {
                if ((length + 1) <= (int)strlen(str_argv))
                {
                    int auto_inc = (str_argv[length] == '=') ? 1 : 0;
                    value = atoi(&str_argv[length + auto_inc]);
                }
                else
                {
                    value = 0;
                }

                found = true;
                continue;
            }
        }
    }

    if (found)
    {
        return value;
    }
    else
    {
        return 0;
    }
}

float get_cmdline_argfloat(const int argc, const char **argv, const char *str_ref)
{
    bool found = false;
    float value = -1;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int start = remove_delimiter_from_string('-', argv[i]);
            const char *str_argv = &argv[i][start];
            int length = (int)strlen(str_ref);

            if (!strncasecmp(str_argv, str_ref, length))
            {
                if ((length + 1) <= (int)strlen(str_argv))
                {
                    int auto_inc = (str_argv[length] == '=') ? 1 : 0;
                    value = (float)atof(&str_argv[length + auto_inc]);
                }
                else
                {
                    value = 0.f;
                }

                found = true;
                continue;
            }
        }
    }

    if (found)
    {
        return value;
    }
    else
    {
        return 0;
    }
}

bool get_cmdline_argstring(const int argc, const char **argv,
                                     const char *str_ref, char **str_retval)
{
    bool found = false;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int start = remove_delimiter_from_string('-', argv[i]);
            char *str_argv = (char *)&argv[i][start];
            int length = (int)strlen(str_ref);

            if (!strncasecmp(str_argv, str_ref, length))
            {
                *str_retval = &str_argv[length+1];
                found = true;
                continue;
            }
        }
    }

    if (!found)
    {
        *str_retval = NULL;
    }

    return found;
}
}//namespace dtb 