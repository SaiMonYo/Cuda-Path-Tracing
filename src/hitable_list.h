#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"
#include "material.cuh"

class hitable_list: public hitable  {
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(hitable **l, int n) {
            list = l;
            list_size = n;
            hitable **temp = new hitable*[n];
            emitters_size = 0;
            
            for (int i = 0; i < n; i++){
                Shapes type = list[i]->type();
                material *mat_ptr;
                if (type == SPHERE){
                    mat_ptr = ((sphere *)list[i])->mat_ptr;
                }
                else if (type == TRIANGLE){
                    mat_ptr = ((triangle *)list[i])->mat_ptr;
                }
                else if (type == QUAD){
                    mat_ptr = ((quad *)list[i])->mat_ptr;
                }
                if (mat_ptr && !is_zero_vector(mat_ptr->emission_strength * mat_ptr->emission_colour)){
                    temp[emitters_size++] = list[i];
                }
            }
            emitters = new hitable*[emitters_size];
            for (int i = 0; i < emitters_size; i++){
                emitters[i] = temp[i];
            }
            delete[] temp;
        }

        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ virtual Shapes type() const;
        __device__ virtual vec3 random_point_on_surface(curandState *local_rand_state) const;
        hitable **list;
        int list_size;
        hitable **emitters;
        int emitters_size;
};

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                temp_rec.obj_ptr = list[i];
                rec = temp_rec;
            }
        }
        return hit_anything;
}

__device__ Shapes hitable_list::type() const{
    return LIST;
}

__device__ vec3 hitable_list::random_point_on_surface(curandState *local_rand_state) const {
    return vec3(-1, -1, -1);
}

#endif