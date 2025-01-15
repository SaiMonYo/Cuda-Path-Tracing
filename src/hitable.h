#pragma once

#include "ray.h"

class material;
class hitable;

enum Shapes{
    EMPTY,
    TRIANGLE,
    SPHERE,
    QUAD
};

struct hit_record{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr = nullptr;
};

class hitable  {
public:
    material *mat_ptr = nullptr;

    __device__ hitable(){}
    __device__ hitable(material* ptr) : mat_ptr(ptr){}
    __device__ virtual Shapes type() const = 0;
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    __device__ virtual vec3 random_point_on_surface(curandState *local_rand_state) const = 0;
    __device__ virtual float area() const = 0;
    __device__ virtual vec3 normal(vec3 point) const = 0;
};