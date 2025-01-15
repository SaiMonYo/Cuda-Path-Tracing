#pragma once

#include "hitable.h"

#define EPSILON 0.000001f

class triangle: public hitable  {
    public:
        vec3 v1, v2, v3;
        vec3 e1, e2;
        vec3 n;
        __device__ triangle() {}
        __device__ triangle(vec3 v_1, vec3 v_2, vec3 v_3, material *m) : v1(v_1), v2(v_2), v3(v_3), hitable(m){
            e1 = v2 - v1;
            e2 = v3 - v1;
            n = unit_vector(cross(e1, e2));
        };
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ virtual Shapes type() const;
        __device__ virtual vec3 random_point_on_surface(curandState *local_rand_state) const;
        __device__ virtual float area() const;
        __device__ virtual vec3 normal(vec3 point) const;
};

__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 h = cross(r.direction(), e2);
    float a = dot(e1, h);

    float f = 1.f / a;
    vec3 s = r.origin() - v1;
    float u = f * dot(s, h);

    if (u >= 0.f && u <= 1.0f){
        vec3 q = cross(s, e1);
        float v = f * dot(r.direction(), q);

        if (v >= 0.f && u + v <= 1.f){
            float t = f * dot(e2, q);

            if (t > t_min && t < t_max){
                rec.t = t;
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = n;
                rec.mat_ptr = mat_ptr;
                return true;
            }
        }
    }
    return false;
}

__device__ Shapes triangle::type() const{
    return TRIANGLE;
}

__device__ vec3 triangle::random_point_on_surface(curandState *local_rand_state) const {
    float u = curand_uniform(local_rand_state);
    float v = curand_uniform(local_rand_state);

    if (u + v >= 1){
        u = 1-u;
        v = 1-v;
    }

    return v1 + e1 * u + e2 * v;
}

__device__ float triangle::area() const{
    return 0.5f * cross(e1, e2).length();
}

__device__ vec3 triangle::normal(vec3 point) const{
    return n;
}