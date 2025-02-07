#include "vec3.h"

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

	inline const char& operator[](int index) const { assert(index >= 0 && index < length); return start[index]; }

    bool operator==(CharHolder other){
        size_t l_length = length;
        size_t r_length = other.length;

        if (l_length != r_length){
            return false;
        }

        for (int i = 0; i < l_length; i++){
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