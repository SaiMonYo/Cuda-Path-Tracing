#include "vec3.h"
#include "charholder.h"


constexpr double FLOAT_DIGITS[] = {0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001};

bool is_whitespace(char c){
    return c == ' ' || c == '\t';
}

bool is_newline(char c){
    return c == '\n' || c == '\r';
}

bool is_digit(char c){
    return '0' <= c && c <= '9';
}

class StringParser : public CharHolder {
public:
    const char *cur = nullptr;

    StringParser(const char *str) : CharHolder(str) { cur = start; }

    template<int N>
    constexpr StringParser(const char (& str)[N]) : CharHolder(str) { cur = start; }
    /**
        advances the cur pointer by n
        default of 1
        returns: value of the char initially pointed at
    */
    char advance(int n = 1);
    /* 
        returns: comparision of a char with the current pointer (safely) 
    */
    bool compare(char c);
    /* 
        returns: comparision with a CharHolder starting at current pointer (safely) 
    */
   bool compare(CharHolder str);
    /*  
        matches a single char,
        advances cur if found 
        returns: the result of the comparision
    */
    bool match(char c);
    /*  matches a string held in a CharHolder,
        advances cur if found to end of the match
        returns: the result of the comparision */
    bool match(CharHolder str);
    /* advances the pointer to the next 
       non-whitespace character */
    void skip_whitespace();
};

char StringParser::advance(int n = 1){
    assert(cur + n < end && "TRYING TO ADVANCE PAST END");

    char c = *cur;

    for (int i = 0; i < n; i++){
        cur++;
    }
    return c;
}

bool StringParser::compare(char c){
    if (cur < end && *cur == c){
        return true;
    }
    return false;
}

bool StringParser::match(char c){
    if (compare(c)){
        advance();
        return true;
    }
    return false;
}

bool StringParser::compare(CharHolder str){
    // check within bounds
    if (cur + str.length >= end){
        return false;
    }
}

bool StringParser::match(CharHolder str){
    if (cur + str.length < end){
        for (int i = 0; i < str.length; i++){
            if (!compare(str[i])){
                return false;
            }
        }
        for (int i = 0; i < str.length; i++){
            advance();
        }
        return true;
    }
    return false;
}