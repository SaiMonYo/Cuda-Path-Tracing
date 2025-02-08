#include "charholder.h"
#include "acutest.h"
#include <assert.h>
#include <string>

const std::string test_strings[] = {"", "a", "abc", "abcdefghi", "aaaa", "d672", "0.", "0.232435", "0.245f"};
const size_t test_strings_lengths[] = {0, 1, 3, 9, 4, 4, 2, 8, 6};

void test_equality(CharHolder ch, std::string str, size_t str_length);
void test_char_ptr_constructor();
void test_char_ptr_start_end_constructor();
void test_strlit_constructor();
void test_ch_ch_equality();
void test_char_ptr_start_end_not_full_constructor();
void test_char_start_and_length();
void test_ch_ch_equality();

TEST_LIST = {
   { "for string literal constructor", test_strlit_constructor },
   { "for single `const char*` argument constructor", test_char_ptr_constructor },
   { "for start and end ptrs arguments of type `const char*` constructor", test_char_ptr_start_end_constructor },
   { "for start and end ptrs arguments of type `const char*` constructor with not full string", test_char_ptr_start_end_constructor },
   { "for start and length arguments of type `const char*`, size_t", test_char_start_and_length },
   { "for equality checks between CharHolders", test_ch_ch_equality },
   { NULL, NULL }     /* zeroed record marking the end of the list */
};


void test_equality(CharHolder ch, std::string str, size_t str_length){
    // check that the lengths are the same for the charholder and str
    // and that ch[index] = str[index] for a known length
    // of the comparision string
    assert(str.length() == str_length);
    
    TEST_CHECK(ch.length == str_length);
    for (size_t i = 0; i < str_length; i++){
        TEST_CHECK(ch[i] == str[i]);
    }
}

void _test_char_ptr_constructor(std::string str){
    size_t str_length = str.length();
    // convert the str to a pointer to the start
    char *start = &str[0];

    // assertion to ensure start is assigned correctly
    assert(start[0] == str[0]);

    CharHolder ch = CharHolder(start);
    test_equality(ch, str, str_length);
}

// wrapper for calling with all data
void test_char_ptr_constructor(){
    for (std::string test: test_strings){
        _test_char_ptr_constructor(test);
    }  
}

void _test_char_ptr_start_end_constructor(std::string str){
    size_t str_length = str.length();
    char *start = &str[0];
    char *end = &str[str_length];

    // assertions for the test
    assert(start[0] == str[0]);

    CharHolder ch = CharHolder(start, end);
    TEST_CHECK(ch.length == str_length);
    test_equality(ch, str, str_length);
}

// wrapper for calling with all data
void test_char_ptr_start_end_constructor(){
    for (std::string test: test_strings){
        _test_char_ptr_start_end_constructor(test);
    }
}

void _test_char_ptr_start_end_not_full_constructor(std::string str){
    size_t str_length = str.length();
    int minus = 0;
    int add = 0;

    if (str_length > 4){
        minus = 2;
        add = 1;
    }
    if (str_length > 3){
        minus = 1;
        add = 1;
    }
    else if (str_length > 2){
        minus = 1;
        add = 0;
    }
    size_t ch_length = str_length - add - minus;

    char *start = &str[add];
    char *end = &str[str_length - minus];

    // assertions for the test
    assert(start[0] == str[add]);
    assert(end[0] == str[str_length - minus]);

    CharHolder ch = CharHolder(start, end);
    TEST_CHECK(ch.length == ch_length);
    test_equality(ch, str.substr(add, ch_length), ch_length);
}

// wrapper for calling with all data
void test_char_ptr_start_end_not_full_constructor(){
    for (std::string test: test_strings){
        _test_char_ptr_start_end_not_full_constructor(test);
    }
}

void _test_char_start_and_length(std::string str){
    size_t str_length = str.length();
    // convert the str to a pointer to the start
    char *start = &str[0];

    // assertion to ensure start is assigned correctly
    assert(start[0] == str[0]);

    CharHolder ch = CharHolder(start, str_length);
    test_equality(ch, str, str_length);

    // test non-start and non-end
    if (str_length > 2){
        ch = CharHolder(start, str_length-1);
        test_equality(ch, str.substr(0,str_length-1), str_length-1);
        ch = CharHolder(start+1, str_length-1);
        test_equality(ch, str.substr(1, str_length-1), str_length-1);
    }
}

// wrapper for calling with all data
void test_char_start_and_length(){
    for (std::string test: test_strings){
        _test_char_start_and_length(test);
    }  
}

void _test_ch_ch_equality_non_full(std::string str, int add, int minus){
    // start from 1 onwards
    size_t str_length = str.length();

    char *start = &str[0];
    char *end = &str[str_length];

    CharHolder ch_start_end_ptr = CharHolder(start+add, end-minus);
    CharHolder ch_start_length = CharHolder(start+add, str_length - minus - add);

    // equals themselves
    TEST_CHECK(ch_start_end_ptr == ch_start_end_ptr);
    TEST_CHECK(ch_start_length == ch_start_length);

    // equals eachother - both directions
    TEST_CHECK(ch_start_length == ch_start_end_ptr);
    TEST_CHECK(ch_start_end_ptr == ch_start_length);

    if (minus != 0){
        return;
    }

    CharHolder ch_start_ptr = CharHolder(start+add);
    TEST_CHECK(ch_start_ptr == ch_start_ptr);

    TEST_CHECK(ch_start_ptr == ch_start_end_ptr);
    TEST_CHECK(ch_start_end_ptr == ch_start_ptr);
    TEST_CHECK(ch_start_ptr == ch_start_length);
    TEST_CHECK(ch_start_length == ch_start_ptr);
}


void _test_ch_ch_equality(std::string str){
    size_t str_length = str.length();

    char *start = &str[0];
    char *end = &str[str_length];

    // assertion to ensure start is assigned correctly
    assert(start[0] == str[0]);

    CharHolder ch_start_ptr = CharHolder(start);
    CharHolder ch_start_end_ptr = CharHolder(start, end);
    CharHolder ch_start_length = CharHolder(start, str_length);

    // equals themselves
    TEST_CHECK(ch_start_ptr == ch_start_ptr);
    TEST_CHECK(ch_start_end_ptr == ch_start_end_ptr);
    TEST_CHECK(ch_start_length == ch_start_length);

    // equals eachother - both directions
    TEST_CHECK(ch_start_ptr == ch_start_end_ptr);
    TEST_CHECK(ch_start_end_ptr == ch_start_ptr);
    TEST_CHECK(ch_start_ptr == ch_start_length);
    TEST_CHECK(ch_start_length == ch_start_ptr);
    TEST_CHECK(ch_start_length == ch_start_end_ptr);
    TEST_CHECK(ch_start_end_ptr == ch_start_length);

    if (str_length <= 2){
        return;
    }
    _test_ch_ch_equality_non_full(str, 1, 0);
    _test_ch_ch_equality_non_full(str, 0, 1);
}

// wrapper for calling with all data
void test_ch_ch_equality(){
    for (std::string test: test_strings){
        _test_ch_ch_equality(test);
    }
    TEST_CHECK(CharHolder("abc") != CharHolder("cba"));
    TEST_CHECK(!(CharHolder("abc") == CharHolder("cba")));
    TEST_CHECK(CharHolder("cba") != CharHolder("abc"));
    TEST_CHECK(!(CharHolder("cba") == CharHolder("abc")));

    TEST_CHECK(CharHolder("abc") != CharHolder("abcd"));
    TEST_CHECK(!(CharHolder("abc") == CharHolder("abcd")));
    TEST_CHECK(CharHolder("abcd") != CharHolder("abc"));
    TEST_CHECK(!(CharHolder("abcd") == CharHolder("abc")));
}

void test_strlit_constructor(){
    // Testing string 0 from the test array
    CharHolder ch = CharHolder("");
    TEST_CHECK(ch.start == ch.end);
    TEST_CHECK(*(ch.start) == '\0');
    TEST_CHECK(ch.length == 0);
    
    // Testting string 1 from the test array
    ch = CharHolder("a");
    test_equality(ch, test_strings[1], test_strings_lengths[1]);
    test_equality(ch, "a", test_strings_lengths[1]);

    // Testting string 2 from the test array
    ch = CharHolder("abc");
    test_equality(ch, test_strings[2], test_strings_lengths[2]);
    test_equality(ch, "abc", test_strings_lengths[2]);

    // Testting string 3 from the test array
    ch = CharHolder("abcdefghi");
    test_equality(ch, test_strings[3], test_strings_lengths[3]);
    test_equality(ch, "abcdefghi", test_strings_lengths[3]);

    // Testting string 4 from the test array
    ch = CharHolder("aaaa");
    test_equality(ch, test_strings[4], test_strings_lengths[4]);
    test_equality(ch, "aaaa", test_strings_lengths[4]);

    // Testting string 5 from the test array
    ch = CharHolder("d672");
    test_equality(ch, test_strings[5], test_strings_lengths[5]);
    test_equality(ch, "d672", test_strings_lengths[5]);

    // Testting string 6 from the test array
    ch = CharHolder("0.");
    test_equality(ch, test_strings[6], test_strings_lengths[6]);
    test_equality(ch, "0.", test_strings_lengths[6]);

    // Testting string 7 from the test array
    ch = CharHolder("0.232435");
    test_equality(ch, test_strings[7], test_strings_lengths[7]);
    test_equality(ch, "0.232435", test_strings_lengths[7]);

    // Testting string 8 from the test array
    ch = CharHolder("0.245f");
    test_equality(ch, test_strings[8], test_strings_lengths[8]);
    test_equality(ch, "0.245f", test_strings_lengths[8]);
}