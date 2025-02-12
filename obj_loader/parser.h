#include "charholder.h"
#include <iostream>


constexpr double FLOAT_DIGITS[] = {0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001};
constexpr size_t FLOAT_DIGITS_LENGTH = 11;

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

    StringParser(CharHolder ch) : CharHolder(ch) { cur = start; }

    template<int N>
    constexpr StringParser(const char (& str)[N]) : CharHolder(str) { cur = start; }
    /**
        advances the cur pointer by n
        default of 1
        returns: value of the char initially pointed at
    */
    char advance(int n);
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
    /* advances the pointer to the next 
       non-whitespace or newline character */
    void skip_whitespace_and_newlines();
    /* parses a string no punctuation
       returns the string at the pointer location*/
    std::string parse_string();
    /* parses an integer value
       returns: the integer value (asserts there was one)*/
    int parse_int();
    /* parses a float value
       returns: the integer value (asserts there was one)*/
    float parse_float();
    /* skips to next line*/
    void next_line();
};

char StringParser::advance(int n = 1){
    assert(cur + n <= end && "TRYING TO ADVANCE PAST END");

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
    assert(str.length > 0);

    // check within bounds
    if (cur + str.length > end){
        return false;
    }


    for (size_t i = 0; i < str.length; i++){
        if (cur[i] != str[i]){
            return false;
        }
    }
    return true;
}

bool StringParser::match(CharHolder str){
    assert(str.length > 0);

    if (!compare(str)) return false;
    
    advance(str.length);
    return true;
}

void StringParser::skip_whitespace(){
    while (cur < end && is_whitespace(*cur)){
        advance();
    }
}

void StringParser::skip_whitespace_and_newlines(){
    while (cur < end && (is_whitespace(*cur) || is_newline(*cur))){
        advance();
    }
}

std::string StringParser::parse_string(){
    std::string ret;
    while (isalnum(*cur)){
        ret += advance();
    }
    return ret;
}

int StringParser::parse_int(){
    bool parsed_digits = false;
    int n = 0;
    while(cur < end && is_digit(*cur)){
        n = 10 * n;
        n = n + (*cur - '0');
        parsed_digits = true;
        advance();
    }
    assert(parsed_digits);

    return n;
}

float StringParser::parse_float(){
    bool parsed_digits = false;

    // integer part
    int integral = 0;

    // check if not in .X form
    if (!compare('.')){
        integral = parse_int();
    }

    // check its not X
    if (!match('.')){
        return (float) integral;
    }

    double fractional = 0;
    size_t index = 0;

    // too many digits for precision
    bool need_to_burn = false;
    while(cur < end && is_digit(*cur)){
        fractional += (*cur - '0') * FLOAT_DIGITS[index++];
        advance();

        parsed_digits = true;

        if (index >= FLOAT_DIGITS_LENGTH){
            need_to_burn = true;
            break;
        }
    }

    assert(parsed_digits);

    // go to end of float
    if (need_to_burn){
        while(cur < end && is_digit(*cur)){
            advance();
        }
    }

    return (float)( ((double)integral) + fractional);
}

void StringParser::next_line(){
    while (cur < end && !is_newline(*cur)){
        advance();
    }
    if (cur < end){
        advance();
    }
}