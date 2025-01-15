#pragma once

#include "hitable.h"

class sphere: public hitable  {
public:
    vec3 center;
    float radius;
    
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), hitable(m)  {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual Shapes type() const;
    __device__ virtual vec3 random_point_on_surface(curandState *local_rand_state) const;
    __device__ virtual float area() const;
    __device__ virtual vec3 normal(vec3 point) const;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            return true;
        }
    }
    return false;
}

__device__ Shapes sphere::type() const{
    return SPHERE;
}

__device__ vec3 sphere::random_point_on_surface(curandState *local_rand_state) const{
    return center + random_in_unit_sphere(local_rand_state) * radius;
}

__device__ float sphere::area() const{
    return 4.0f * M_PI * radius * radius;
}

__device__ vec3 sphere::normal(vec3 point) const{
    return (point - center) / radius;
}