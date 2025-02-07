#pragma once

struct hit_record;

#include "ray.h"
#include "hitable.h"

__device__ float schlick(float cosine, float ref_idx);
__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted);
__device__ vec3 reflect(const vec3& v, const vec3& n);
__device__ float fresnel_reflectance(float cos_i, float ior);
__device__ float roughness_to_alpha(float linear_roughness);
__device__ float ggx_D(vec3 micro_normal, float alpha_x, float alpha_y);
__device__ float ggx_lambda(vec3 omega, float alpha_x, float alpha_y);
__device__ float ggx_G1(vec3 omega, float alpha_x, float alpha_y);
__device__ float ggx_G2(vec3 omega_o, vec3 omega_i, vec3 omega_m, float alpha_x, float alpha_y);
__device__ vec3 sample_visible_normals_ggx(vec3 omega, float alpha_x, float alpha_y, curandState *local_rand_state);


class material  {
public:
    vec3 albedo;
    float emission_strength;
    vec3 emission_colour;
    
    __device__ material(const vec3& a, float em_s, const vec3& em_c) : albedo(a), emission_strength(em_s), emission_colour(em_c){}
    __device__ virtual bool sample(const hit_record& rec, const ray &r_in, vec3 &throughput, ray &scattered, float& pdf, curandState *local_rand_state) const = 0;
    __device__ virtual bool eval(const hit_record& rec, const vec3& w_i, const vec3& w_o, float& pdf, vec3& brdf, curandState *local_rand_state) const = 0;
};


class lambertian : public material {
public:
    __device__ lambertian(const vec3& a) : material(a,0.0f,vec3(0.0f)) {}
    __device__ lambertian(const vec3& a, float em_s, const vec3& em_c) : material(a,em_s,em_c) {}
    __device__ virtual bool sample(const hit_record& rec, const ray &r_in, vec3 &throughput, ray &scattered, float& pdf, curandState *local_rand_state) const {
        vec3 direction = random_cosine_weighted_direction(local_rand_state);
        scattered = ray(rec.p, local_to_global(direction, rec.normal));
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

__device__ float fresnel_reflectance(float cos_i, float ior){
    float eta1 = 1.0f;
    float eta2 = ior;

    if (cos_i < 0.0f){
        float temp = eta1;
        eta1 = eta2;
        eta2 = temp;
    }

    float eta = eta1 / eta2;
    float sint_sq = eta * eta * (1 - cos_i * cos_i);

    if (sint_sq >= 1.0f){
        return 1.0f;
    }

    float cos_t = sqrtf(1.0f - sint_sq);

    float R_s = powf((eta1 * cos_i - eta2 * cos_t) / (eta1 * cos_i + eta2 * cos_t), 2.0f);
    float R_p = powf((eta1 * cos_t - eta2 * cos_i) / (eta1 * cos_t + eta2 * cos_i), 2.0f);

    return 0.5 * (R_s + R_p);
}

__device__ inline float roughness_to_alpha(float linear_roughness) {
	return fmaxf(1e-6f, square(linear_roughness));
}

__device__ float ggx_D(vec3 micro_normal, float alpha_x, float alpha_y){
    if (micro_normal.z < 1e-6f) {
		return 0.0f;
	}

	float sx = -micro_normal.x / (micro_normal.z * alpha_x);
	float sy = -micro_normal.y / (micro_normal.z * alpha_y);

	float sl = 1.0f + sx * sx + sy * sy;

	float cos_theta_2 = micro_normal.z * micro_normal.z;
	float cos_theta_4 = cos_theta_2 * cos_theta_2;

	return 1.0f / (sl * sl * M_PI * alpha_x * alpha_y * cos_theta_4);
}


__device__ float ggx_lambda(vec3 omega, float alpha_x, float alpha_y){
    return 0.5f * (sqrtf(1.0f + (square(alpha_x * omega.x) + square(alpha_y * omega.y)) / square(omega.z)) - 1.0f);
}

__device__ float ggx_G1(vec3 omega, float alpha_x, float alpha_y){
    return 1.0f / (1.0f + ggx_lambda(omega, alpha_x, alpha_y));
}


__device__ float ggx_G2(vec3 omega_o, vec3 omega_i, vec3 omega_m, float alpha_x, float alpha_y){
    bool omega_i_backfacing = dot(omega_i, omega_m) * omega_i.z <= 0.0f;
	bool omega_o_backfacing = dot(omega_o, omega_m) * omega_o.z <= 0.0f;

	if (omega_i_backfacing || omega_o_backfacing) {
		return 0.0f;
	} else {
		return 1.0f / (1.0f + ggx_lambda(omega_o, alpha_x, alpha_y) + ggx_lambda(omega_i, alpha_x, alpha_y));
	}
}

__device__ vec3 sample_visible_normals_ggx(vec3 omega, float alpha_x, float alpha_y, curandState *local_rand_state){
	// Transform the view direction to the hemisphere configuration
	vec3 v = normalise(vec3(alpha_x * omega.x, alpha_y * omega.y, omega.z));

	// Orthonormal basis (with special case if cross product is zero)
	float length_squared = v.x*v.x + v.y*v.y;
	vec3 axis_1 = length_squared > 0.0f ? vec3(-v.y, v.x, 0.0f) / sqrtf(length_squared) : vec3(1.0f, 0.0f, 0.0f);
	vec3 axis_2 = cross(v, axis_1);

	// Parameterization of the projected area
	vec3 d = random_in_unit_disk(local_rand_state);
	float t1 = d.x;
	float t2 = lerp(safe_sqrt(1.0f - t1*t1), d.y, 0.5f + 0.5f * v.z);

	// Reproject onto hemisphere
	vec3 n_h = t1*axis_1 + t2*axis_2 + safe_sqrt(1.0f - t1*t1 - t2*t2) * v;

	// Transform the normal back to the ellipsoid configuration
	return normalise(vec3(alpha_x * n_h.x, alpha_y * n_h.y, n_h.z));
}





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