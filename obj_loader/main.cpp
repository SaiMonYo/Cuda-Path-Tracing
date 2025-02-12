#include "parser.h"
#include <iostream>



int main(){
    StringParser s = StringParser("help me ahahahahha");
    std::cout << s.parse_string() << std::endl;
    s.skip_whitespace();
    std::cout << s.parse_string() << std::endl;
    s.skip_whitespace();
    std::cout << s.parse_string() << std::endl;
    return 0;
}