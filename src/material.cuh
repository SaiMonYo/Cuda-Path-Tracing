#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"


__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0f*dot(v,n)*n;
}




class material  {
    public:
        vec3 colour;
        float emission_strength;
        vec3 emission_colour;
        __device__ material(const vec3& c, float em_s, const vec3& em_c) : colour(c), emission_strength(em_s), emission_colour(em_c){}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, ray& scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material {
    public:
        __device__ lambertian(const vec3& a) : material(a,0.0f,vec3(0.0f)) {}
        __device__ lambertian(const vec3& a, float em_s, const vec3& em_c) : material(a,em_s,em_c) {}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, ray& scattered, curandState *local_rand_state) const  {
            vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            scattered = ray(rec.p, target-rec.p);
            return true;
        }

};

class metal : public material {
    public:
        __device__ metal(const vec3& a, float f) : material(a,0.0f,vec3(0.0f)) { if (f < 1) fuzz = f; else fuzz = 1; }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, ray& scattered, curandState *local_rand_state) const  {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));

            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }
        float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri), material(vec3(1.f),0.0f,vec3(0.0f)) {}
    __device__ virtual bool scatter(const ray& r_in,
                         const hit_record& rec,
                         ray& scattered,
                         curandState *local_rand_state) const  {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};
#endif