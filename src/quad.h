#pragma once

#include "triangle.h"

#define EPSILON 0.000001f

class quad: public hitable  {
public:
    triangle a, b;
    vec3 e1, e2;
    vec3 v1;
    vec3 n;
    
    __device__ quad() {}
    __device__ quad(vec3 v_1, vec3 v_2, vec3 v_3, vec3 v_4, material *m) : 
                    a(v_1, v_2, v_3, nullptr), b(v_1, v_3, v_4, nullptr), v1(v_1), hitable(m) {
                        e1 = v_2-v_1;
                        e2 = v_4-v_1;
                        n = unit_vector(cross(e1, e2));
                    };
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual Shapes type() const;
    __device__ virtual vec3 random_point_on_surface(curandState *local_rand_state) const;
    __device__ virtual float area() const;
    __device__ virtual vec3 normal(vec3 point) const;
};

__device__ bool quad::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    return a.hit(r, t_min, t_max, rec) || b.hit(r, t_min, t_max, rec);
}


__device__ Shapes quad::type() const{
    return QUAD;
}

__device__ vec3 quad::random_point_on_surface(curandState *local_rand_state) const{
    float u = curand_uniform(local_rand_state);
    float v = curand_uniform(local_rand_state);

    return v1 + e1 * u + e2 * v;
}

__device__ float quad::area() const{
    return cross(e1, e2).length();
}

__device__ vec3 quad::normal(vec3 point) const{
    return n;
}