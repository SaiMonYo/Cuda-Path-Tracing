#include "charholder.h"
#include "acutest.h"
#include <assert.h>
#include <string>
#include "parser.h"
#include <cmath>

void run_tests();

TEST_LIST = {
    { "For testing functionality of parser", run_tests },
    { NULL, NULL }     /* zeroed record marking the end of the list */
};


inline bool float_equals(float a, float b){
    return fabs(b - a) < 0.002f;
}


void run_tests(){
    StringParser parser = StringParser("hello");
    TEST_CHECK(parser.advance() == 'h');
    TEST_CHECK(parser.advance() == 'e');
    TEST_CHECK(parser.advance() == 'l');
    TEST_CHECK(parser.advance() == 'l');
    TEST_CHECK(parser.advance() == 'o');
    TEST_CHECK(parser.end - parser.cur == 0);

    parser = StringParser("test");
    TEST_CHECK(parser.compare('t'));
    TEST_CHECK(!parser.compare('x'));


    parser = StringParser("hello");
    CharHolder str = CharHolder("hello");
    CharHolder str2 = CharHolder("world");
    TEST_CHECK(parser.compare(str));
    TEST_CHECK(!parser.compare(str2));

    parser = StringParser("hello");
    TEST_CHECK(parser.match('h'));
    TEST_CHECK(parser.cur - parser.start == 1);
    TEST_CHECK(!parser.match('x'));

    parser = StringParser("hello");
    str = CharHolder("hello");
    str2 = CharHolder("hell");
    TEST_CHECK(parser.match(str2));
    TEST_CHECK(!parser.match(CharHolder("world")));

    parser = StringParser("   abc");
    parser.skip_whitespace();
    TEST_CHECK(parser.compare('a'));

    parser = StringParser(" \n \n abc");
    parser.skip_whitespace_and_newlines();
    TEST_CHECK(parser.compare('a'));

    parser = StringParser("1234 abc");
    TEST_CHECK(parser.parse_int() == 1234);
    parser.skip_whitespace();
    TEST_CHECK(parser.compare('a'));

    parser = StringParser("3.14 test");
    TEST_CHECK(float_equals(parser.parse_float(), 3.14f));
    parser.skip_whitespace();
    TEST_CHECK(parser.compare('t'));
}