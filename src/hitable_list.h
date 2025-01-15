#pragma once

#include "hitable.h"
#include "material.cuh"
#include "assert.h"


class hitable_list{
public:
    __device__ hitable_list() {}
    __device__ hitable_list(hitable **l, int n);
    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __host__ __device__ inline hitable* operator[](int i) const { return list[i]; }
    __device__ hitable* get_random_emitter(curandState *local_rand_state);
    __device__ ~hitable_list() {delete[] emitters; delete[] list;}

    hitable **list;
    int list_size;
    hitable **emitters;
    int emitters_size;
};

__device__ hitable_list::hitable_list(hitable **l, int n){
    list = l;
    list_size = n;
    hitable **temp = new hitable*[n];
    emitters_size = 0;
    
    for (int i = 0; i < n; i++){
        material *mat_ptr = list[i]->mat_ptr;
        if (mat_ptr != nullptr && mat_ptr->emission_strength > 0.0f){
            temp[emitters_size++] = list[i];
        }
    }
    emitters = new hitable*[emitters_size];
    for (int i = 0; i < emitters_size; i++){
        emitters[i] = temp[i];
    }
    delete[] temp;
}

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ hitable* hitable_list::get_random_emitter(curandState *local_rand_state){
    assert(emitters_size > 0);
    int idx = curand(local_rand_state) % emitters_size;
    return emitters[idx];
}