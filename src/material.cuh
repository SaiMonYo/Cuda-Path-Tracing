#pragma once

struct hit_record;

#include "ray.h"
#include "hitable.h"

class material  {
public:
    vec3 albedo;
    float emission_strength;
    vec3 emission_colour;
    
    __device__ material(const vec3& a, float em_s, const vec3& em_c) : albedo(a), emission_strength(em_s), emission_colour(em_c){}
    __device__ virtual bool sample(const hit_record& rec, const ray &r_in, vec3 &throughput, ray &scattered, float& pdf, curandState *local_rand_state) const = 0;
    __device__ virtual bool eval(const hit_record& rec, const vec3& w_i, const vec3& w_o, float& pdf, vec3& brdf, curandState *local_rand_state) const = 0;
};


__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = normalise(v);
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

__device__ float fresnel_reflectance(const hit_record &rec, vec3 w_o, float ior){
    float eta1 = 1.0f;
    float eta2 = ior;

    if (dot(rec.normal, w_o) < 0.0f){
        float temp = eta1;
        eta1 = eta2;
        eta2 = temp;
    }

    float eta = eta1 / eta2;
    float cos_i = dot(w_o, rec.normal);
    float sint_sq = eta * eta * (1 - cos_i * cos_i);

    if (sint_sq >= 1.0f){
        return 1.0f;
    }

    float cos_t = sqrtf(1.0f - sint_sq);

    float R_s = powf((eta1 * cos_i - eta2 * cos_t) / (eta1 * cos_i + eta2 * cos_t), 2.0f);
    float R_p = powf((eta1 * cos_t - eta2 * cos_i) / (eta1 * cos_t + eta2 * cos_i), 2.0f);

    return 0.5 * (R_s + R_p);
}


class lambertian : public material {
public:
    __device__ lambertian(const vec3& a) : material(a,0.0f,vec3(0.0f)) {}
    __device__ lambertian(const vec3& a, float em_s, const vec3& em_c) : material(a,em_s,em_c) {}
    __device__ virtual bool sample(const hit_record& rec, const ray &r_in, vec3 &throughput, ray &scattered, float& pdf, curandState *local_rand_state) const {
        vec3 direction = random_cosine_weighted_direction(local_rand_state);
        scattered = ray(rec.p, orientate_hemisphere_to_other(direction, rec.normal));
        pdf = direction.z * ONE_OVER_PI;
        throughput *= albedo;
        return true;
    }

    __device__ virtual bool eval(const hit_record& rec, const vec3& w_i, const vec3& w_o, float& pdf, vec3& brdf, curandState *local_rand_state) const{
        float cos_i = dot(w_i, rec.normal);
        if (cos_i < 0.0f){
            return false;
        }
        brdf = albedo * cos_i * ONE_OVER_PI;
        pdf = cos_i * ONE_OVER_PI;
        return true;
    }
};


class plastic : public material {
public:
    float ior;
    float eta_sq;
    float R_i;

    __device__ plastic(float ior, vec3 colour) : ior(ior), material(colour, 0.0f, vec3(0.0f)) {
        float n_i = ior;
        float n_i2 = n_i * n_i;
        float n_i4 = n_i2 * n_i2;

        float R_e = 0.5f 
            + ((n_i-1)*(3.0f*n_i + 1) / (6.0f * powf((n_i + 1), 2.0f)))
            + ((n_i2 * powf(n_i2 - 1.0f, 2.0f)) / (powf(n_i2 + 1.0f, 3.0f))) * logf((n_i - 1.0f) / (n_i + 1.0f))
            - (((2.0f*n_i2*n_i) * (n_i2 + 2.0f*n_i - 1.0f)) / ((n_i2 + 1)*(n_i4 - 1.0f)))
            + ((8.0f * n_i4 * (n_i4 + 1.0f)) / ((n_i2 + 1.0f) * powf(n_i4 - 1.0f, 2.0f))) * logf(n_i);
        
        R_i = 1.0f - (1.0f /  n_i2) * (1.0f - R_e);
        eta_sq = powf(1.0f / ior, 2.0f);
    }

    __device__ virtual bool sample(const hit_record& rec, const ray &r_in, vec3 &throughput, ray &scattered, float& pdf, curandState *local_rand_state) const {
        vec3 w_o = -r_in.direction;
        float F_i = fresnel_reflectance(rec, w_o, ior);

        vec3 w_i;
        if (curand_uniform(local_rand_state) > F_i){
            vec3 local_w_i = random_cosine_weighted_direction(local_rand_state);
            w_i = orientate_hemisphere_to_other(local_w_i, rec.normal);

        }
        else{
            w_i = reflect(r_in.direction, rec.normal);
            pdf = F_i;
        }
        float F_o = fresnel_reflectance(rec, w_i, ior);

        float cos_i = fmaxf(0.0f, dot(w_i, rec.normal));

        vec3 brdf_specular = vec3(F_i);
        vec3 brdf_diffuse = eta_sq * (1.0f-F_i) * albedo * ONE_OVER_PI * (1.0f - F_o) / (1.0f - R_i) * cos_i;

        float pdf_diffuse = (1-F_i) * cos_i * ONE_OVER_PI;
        float pdf_specular = F_i;
        pdf = lerp(pdf_diffuse, pdf_specular, F_i);

        throughput *= (brdf_specular + brdf_diffuse) / pdf;

        scattered = ray(rec.p, w_i);

        return true;
    }

    __device__ virtual bool eval(const hit_record& rec, const vec3& w_i, const vec3& w_o, float& pdf, vec3& brdf, curandState *local_rand_state) const{
        float F_i = fresnel_reflectance(rec, w_o, ior);
        float F_o = fresnel_reflectance(rec, w_i, ior);

        float cos_i = fmaxf(0.0f, dot(w_i, rec.normal));

        vec3 brdf_specular = vec3(F_i);
        vec3 brdf_diffuse = eta_sq * (1.0f-F_i) * albedo * ONE_OVER_PI * (1.0f - F_o) / (1.0f - R_i) * cos_i;

        float pdf_diffuse = (1-F_i) * cos_i * ONE_OVER_PI;
        float pdf_specular = F_i;

        brdf = brdf_diffuse + brdf_specular;

        pdf = lerp(pdf_diffuse, pdf_specular, F_i);

        return isfinite(pdf) && pdf > 0.0f;
    } 
};











/*
class metal : public material {
public:
    vec3 albedo;
    float fuzz;

    __device__ metal(const vec3& a, float f) : material(a,0.0f,vec3(0.0f)) { if (f < 1) fuzz = f; else fuzz = 1;}
    __device__ virtual bool sample(const hit_record& rec, const ray &r_in, vec3 &throughput, ray &scattered, curandState *local_rand_state) const  {
        vec3 reflected = reflect(normalise(r_in.direction), rec.normal);
        scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
        return (dot(scattered.direction, rec.normal) > 0.0f);
    }


    __device__ virtual bool eval(const hit_record& rec, const vec3& w_i, const vec3& w_o, curandState *local_rand_state) const{
        return false;
    }
};

class dielectric : public material {
public:
    float ref_idx;

    __device__ dielectric(float ri) : ref_idx(ri), material(vec3(1.f),0.0f,vec3(0.0f)) {}
    __device__ virtual bool sample(const hit_record& rec, const ray &r_in, vec3 &throughput, ray &scattered, curandState *local_rand_state) const  {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction, rec.normal);
        float ni_over_nt;
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction, rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction, rec.normal) / length(r_in.direction);
            cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction, rec.normal) / length(r_in.direction);
        }
        if (refract(r_in.direction, outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    __device__ virtual bool eval(const hit_record& rec, const vec3& w_i, const vec3& w_o, curandState *local_rand_state) const{
        return false;
    }
};
*/