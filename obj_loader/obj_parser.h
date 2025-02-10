#include "parser.h"
#include "base_triangle.h"
#include <string>
#include <fstream>
#include <vector>


CharHolder load_file_contents(std::string filename){
    std::ifstream t;
    int length;
    t.open(filename);
    t.seekg(0, std::ios::end);
    length = t.tellg();
    t.seekg(0, std::ios::beg);
    char *buffer = new char[length];
    t.read(buffer, length);
    t.close();

    return CharHolder(buffer, length);
};

BaseTriangle **parse_obj(std::string filename){
    std::vector<BaseTriangle> triangles;
    
    CharHolder file_contents = load_file_contents(filename);

    StringParser parser = StringParser(file_contents);

    free((char *)file_contents.start);
};