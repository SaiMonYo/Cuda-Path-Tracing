#pragma once

#include "triangle.h"

#define EPSILON 0.000001f

class quad: public hitable  {
    public:
        material *mat_ptr;
        triangle a, b;
        __device__ quad() {}
        __device__ quad(vec3 v_1, vec3 v_2, vec3 v_3, vec3 v_4, material *m) : 
                        a(v_1, v_2, v_3, m), 
                        b(v_1, v_3, v_4, m), 
                        mat_ptr(m) {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ virtual Shapes type() const;
        __device__ void clean();
};

__device__ bool quad::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    return a.hit(r, t_min, t_max, rec) || b.hit(r, t_min, t_max, rec);
}


__device__ Shapes quad::type() const{
    return QUAD;
}