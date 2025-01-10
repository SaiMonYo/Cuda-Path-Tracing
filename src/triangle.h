#pragma once

#include "hitable.h"

#define EPSILON 0.000001f

class triangle: public hitable  {
    public:
        material *mat_ptr;
        vec3 v1, v2, v3;
        __device__ triangle() {}
        __device__ triangle(vec3 v_1, vec3 v_2, vec3 v_3, material *m) : v1(v_1), v2(v_2), v3(v_3), mat_ptr(m){};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ virtual Shapes type() const;
};

__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 v1v2 = v2-v1;
    vec3 v1v3 = v3-v1;
    
    vec3 h = cross(r.direction(), v1v3);
    float a = dot(v1v2, h);

    float f = 1.f / a;
    vec3 s = r.origin() - v1;
    float u = f * dot(s, h);

    if (u >= 0.f && u <= 1.0f){
        vec3 q = cross(s, v1v2);
        float v = f * dot(r.direction(), q);

        if (v >= 0.f && u + v <= 1.f){
            float t = f * dot(v1v3, q);

            if (t > t_min && t < t_max){
                rec.t = t;
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = unit_vector(cross(v1v2, v1v3));
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