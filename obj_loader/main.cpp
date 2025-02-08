#include "parser.h"
#include <iostream>



int main(){
    StringParser s = StringParser("0.77739212334");
    std::cout << s.parse_float() << std::endl;
    return 0;
}