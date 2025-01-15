#pragma once

#include "vec3.h"

class ray{
public:
    vec3 origin, direction;

    __device__ ray() {}
    __device__ ray(const vec3& o, const vec3& dir) : origin(o), direction(dir) {}
    __device__ vec3 at(float t) const { return origin + t  * direction; }
};