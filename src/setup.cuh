#pragma once

#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "triangle.h"
#include "quad.h"
#include "hitable_list.h"
#include "camera.cuh"
#include "material.cuh"

#define white vec3(.82f)
#define red vec3(.73f,0.05f,0.05f)
#define green vec3(0.12f,.64f,0.15f)

#define RND (curand_uniform(&local_rand_state))


__global__ void create_cornell(hitable **d_list, hitable_list** hit_list, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;
        d_list[i++] = new quad(vec3(552.8f, 0.f, 0.f), vec3(0.f, 0.f, 0.f), vec3(0.f, 0.f, 559.2f), vec3(549.6f, 0.f, 559.2f), new lambertian(white));
        d_list[i++] = new quad(vec3(343.0f, 548.75f, 227.0f), vec3(343.0f, 548.75f, 332.0f), vec3(213.0f, 548.75f, 332.0f), vec3(213.0f, 548.75f, 227.0f), new lambertian(vec3(1.0f), 15.0f, vec3(1.0f)));
        d_list[i++] = new quad(vec3(556.0f, 548.8f, 0.0f), vec3(556.0f, 548.8f, 559.2f), vec3(0.0f, 548.8f, 559.2f), vec3(0.0f, 548.8f, 0.0f), new lambertian(white));
        d_list[i++] = new quad(vec3(549.6f, 0.0f, 559.2f), vec3(0.0f, 0.0f, 559.2f), vec3(0.0f, 548.8f, 559.2f), vec3(556.0f, 548.8f, 559.2f), new lambertian(white));
        d_list[i++] = new quad(vec3(0.0f, 0.0f, 559.2f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 548.8f, 0.0f), vec3(0.0f, 548.8f, 559.2f), new lambertian(green));
        d_list[i++] = new quad(vec3(552.8f, 0.0f, 0.0f), vec3(549.6f, 0.0f, 559.2f), vec3(556.0f, 548.8f, 559.2f), vec3(556.0f, 548.8f, 0.0f), new lambertian(red));
        d_list[i++] = new quad(vec3(130.0f, 165.0f, 65.0f), vec3(82.0f, 165.0f, 225.0f), vec3(240.0f, 165.0f, 272.0f), vec3(290.0f, 165.0f, 114.0f), new lambertian(white));
        d_list[i++] = new quad(vec3(290.0f, 0.0f, 114.0f), vec3(290.0f, 165.0f, 114.0f), vec3(240.0f, 165.0f, 272.0f), vec3(240.0f, 0.0f, 272.0f), new lambertian(white));
        d_list[i++] = new quad(vec3(130.0f, 0.0f, 65.0f), vec3(130.0f, 165.0f, 65.0f), vec3(290.0f, 165.0f, 114.0f), vec3(290.0f, 0.0f, 114.0f), new lambertian(white));
        d_list[i++] = new quad(vec3(82.0f, 0.0f, 225.0f), vec3(82.0f, 165.0f, 225.0f), vec3(130.0f, 165.0f, 65.0f), vec3(130.0f, 0.0f, 65.0f), new lambertian(white));
        d_list[i++] = new quad(vec3(240.0f, 0.0f, 272.0f), vec3(240.0f, 165.0f, 272.0f), vec3(82.0f, 165.0f, 225.0f), vec3(82.0f, 0.0f, 225.0f), new lambertian(white));
        d_list[i++] = new quad(vec3(423.0f, 330.0f, 247.0f), vec3(265.0f, 330.0f, 296.0f), vec3(314.0f, 330.0f, 456.0f), vec3(472.0f, 330.0f, 406.0f), new lambertian(white));
        d_list[i++] = new quad(vec3(423.0f, 0.0f, 247.0f), vec3(423.0f, 330.0f, 247.0f), vec3(472.0f, 330.0f, 406.0f), vec3(472.0f, 0.0f, 406.0f), new lambertian(white));
        d_list[i++] = new quad(vec3(472.0f, 0.0f, 406.0f), vec3(472.0f, 330.0f, 406.0f), vec3(314.0f, 330.0f, 456.0f), vec3(314.0f, 0.0f, 456.0f), new lambertian(white));
        d_list[i++] = new quad(vec3(314.0f, 0.0f, 456.0f), vec3(314.0f, 330.0f, 456.0f), vec3(265.0f, 330.0f, 296.0f), vec3(265.0f, 0.0f, 296.0f), new lambertian(white));
        d_list[i++] = new quad(vec3(265.0f, 0.0f, 296.0f), vec3(265.0f, 330.0f, 296.0f), vec3(423.0f, 330.0f, 247.0f), vec3(423.0f, 0.0f, 247.0f), new lambertian(white));
        d_list[i++] = new sphere(vec3(400.0f, 100.0f, 200.0f), 100.0f, new plastic(1.5f, vec3(1.0f,1.0f,0.0f)));
        //d_list[i++] = new sphere(vec3(400.0f, 100.0f, 200.0f), 100.0f, new lambertian(vec3(1.0f,1.0f,0.0f)));

        *hit_list = new hitable_list(d_list, 17);

        vec3 lookfrom(278.f,273.f,-800.f);
        vec3 lookat(278.f,273.f,-799.f);
        float dist_to_focus = length(lookfrom - vec3(278, 273, 280));
        float aperture = 0.025f;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 40.0f,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable_list **d_world, camera **d_camera, int num_hitables) {
    for(int i=0; i < num_hitables; i++) {
        delete d_list[i]->mat_ptr;
        delete d_list[i];
    }
    delete d_world;
    delete *d_camera;
}


/*
__global__ void create_world(hitable **d_list, hitable_list** hit_list, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0f,-1), 1000,
                               new lambertian(vec3(0.5f, 0.5f, 0.5f)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2f,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2f,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2f,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2f, new dielectric(1.5f));
                }
            }
        }

        d_list[i++] = new sphere(vec3(-20, 20, 0), 10.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f), 1.0f, vec3(0.f, 0.f, 1.f)));
        d_list[i++] = new triangle(vec3(-2.f, 1.f, -4.8f),vec3(0.f, 2.f, -5.2f),vec3(1.f, 1.f, -5.f), new lambertian(vec3(1), 1.f, vec3(1.f, 0, 0)));
        d_list[i++] = new sphere(vec3(4, 10, 0),  1.0f, new lambertian(vec3(0.7f, 0.6f, 0.5f), 1.0f, vec3(1)));
        *rand_state = local_rand_state;
        *hit_list = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,3,2);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10;length(lookfrom-lookat);
        float aperture = 0.1f;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0f,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}
*/