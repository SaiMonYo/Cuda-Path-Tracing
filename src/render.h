#pragma once

#include "setup.cuh"

namespace Sample{

__device__ vec3 NEE(const ray& r, hitable_list **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 radiance = 0.f;
    vec3 throughput = 1.f;

    for (int i = 0; i <= 50; i++){
        hit_record rec;
        if (!(*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            break;
        }
        material *mat = rec.mat_ptr;
        
        if (mat->emission_strength > 0.0f){
            if (i==0){
                radiance = throughput * mat->emission_colour * mat->emission_strength;
            }
            break;
        }


        // pick random emitter to sample 
        hitable *random_emitter = (*world)->get_random_emitter(local_rand_state);
        vec3 random_point_on_emitter = random_emitter->random_point_on_surface(local_rand_state);

        float emitter_area = random_emitter->area();

        vec3 to_light = random_point_on_emitter - rec.p;
        float distance_to_light_squared = length_squared(to_light);
        float distance_to_light = sqrtf(distance_to_light_squared);

        to_light = to_light / distance_to_light;

        // calculate the normal of the emitter at the random sampled point
        vec3 light_normal = random_emitter->normal(random_point_on_emitter);

        float cos_o = -dot(to_light, light_normal);
        float cos_i = dot(to_light, rec.normal);


        // the light can contribute to the hit location
        if (cos_o > 0.0f && cos_i > 0.0f){
            // trace shadow ray to the point to see if it can cantribute
            ray shadow_ray = ray(rec.p, to_light);
            hit_record shadow_rec;
            if (!(*world)->hit(shadow_ray, 0.001f, distance_to_light - 0.001f, shadow_rec)){
                float solid_angle = (cos_o * emitter_area) / distance_to_light_squared;

                float pdf;
                vec3 brdf;
                if(mat->eval(rec, to_light, -cur_ray.direction, pdf, brdf, local_rand_state)){
                    material *light_mat = random_emitter->mat_ptr;
                    vec3 emitter_colour = light_mat->emission_colour * light_mat->emission_strength;

                    // weight this sample
                    radiance += throughput * brdf * emitter_colour * solid_angle;
                }
            }
        }

        ray scattered;
        float pdf;
        // get the scattered ray
        if(rec.mat_ptr->sample(rec, cur_ray, throughput, scattered, pdf, local_rand_state)) {
            cur_ray = scattered;
        }
        else{
            break;
        }


        // apply russian roulette if theres been 4 bounces
        if (i >= 4){
            float one_minus_p = max_component(throughput);

            if (curand_uniform(local_rand_state) > one_minus_p){
                break;
            }
            // weight throughput if it wasnt culled
            throughput /= one_minus_p;
        }

    }
    return radiance;
}

__device__ vec3 naive(const ray& r, hitable_list **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 radiance = 0.f;
    vec3 throughput = 1.f;

    for (int i = 0; i <= 50; i++){
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            material *mat = rec.mat_ptr;
            vec3 emitted_light = mat->emission_colour * mat->emission_strength;
            radiance += emitted_light * throughput;
            ray scattered;
            float pdf;
            if(rec.mat_ptr->sample(rec, cur_ray, throughput, scattered, pdf, local_rand_state)) {
                cur_ray = scattered;
            }

            float p = max_component(throughput);
            if (curand_uniform(local_rand_state) >= p){
                break;
            }
            throughput *= 1.f / p;
        }
        else{
            break;
        }
    }
    return radiance;
}

}