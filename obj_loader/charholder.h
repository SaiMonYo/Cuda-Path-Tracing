#pragma once

#include <cstddef>
#include <assert.h>
#include <string>
#include <cstring>

class CharHolder{
public:
    const char *start = nullptr;
    const char *end = nullptr;
    size_t length = -1;

    template<int N>
    CharHolder(const char (& str)[N]){
        start = str;
        end = str + N - 1;
        length = N;
    }

    CharHolder(const char *str){
        start = str;
        length = strlen(str);
        end = str + length;
    }

    CharHolder(const char *str, size_t len){
        start = str;
        length = len;
        end = str + length;
    }

    CharHolder(const char *s, const char *e){
        start = s;
        end = e;
        length = end - start;
    }

    CharHolder(const CharHolder& ch){
        start = ch.start;
        end = ch.end;
        length = end - start;
    }

	inline const char& operator[](int index) { assert(index >= 0 && (unsigned)index < length); return start[index]; }
    inline const char operator[](int index) const { assert(index >= 0 && (unsigned)index < length); return start[index]; }

    bool operator==(CharHolder other){
        size_t l_length = length;
        size_t r_length = other.length;

        if (l_length != r_length){
            return false;
        }

        for (size_t i = 0; i < l_length; i++){
            if ((*this)[i] != other[i]){
                return false;
            }
        }
        return true;
    }

    bool operator!=(CharHolder other){
        return !(operator==(other));
    }
};