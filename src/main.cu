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

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 incoming_light = 0.;
    vec3 ray_colour = 1.f;

    for (int i = 0; i <= 50; i++){
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            if(rec.mat_ptr->scatter(cur_ray, rec, scattered, local_rand_state)) {
                cur_ray = scattered;
            }
            //return rec.normal;
            material *mat = rec.mat_ptr;
            vec3 emitted_light = mat->emission_colour * mat->emission_strength;
            float light_strength = dot(rec.normal, r.direction());
            incoming_light += emitted_light * ray_colour;
            ray_colour *= mat->colour;

            float p = fmax(ray_colour.x(), fmax(ray_colour.y(), ray_colour.z()));

            if (curand_uniform(local_rand_state) >= p){
                break;
            }
            ray_colour *= 1.f / p;
        }
        else{
            break;
        }
    }
    return incoming_light;
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }

        d_list[i++] = new sphere(vec3(-20, 20, 0), 10.0, new lambertian(vec3(0.4, 0.2, 0.1), 1.0f, vec3(0.f, 0.f, 1.f)));
        d_list[i++] = new triangle(vec3(-2.f, 1.f, -4.8f),vec3(0.f, 2.f, -5.2f),vec3(1.f, 1.f, -5.f), new lambertian(vec3(1), 1.f, vec3(1.f, 0, 0)));
        d_list[i++] = new sphere(vec3(4, 10, 0),  1.0, new lambertian(vec3(0.7, 0.6, 0.5), 1.0f, vec3(1)));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,3,2);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10;(lookfrom-lookat).length();
        float aperture = 0.1f;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

#define white vec3(.73f)
#define red vec3(.65f,0.05f,0.05f)
#define green vec3(0.12f,.45f,0.15f)


__global__ void create_cornell(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;
        d_list[i++] = new quad(vec3(552.8f, 0.f, 0.f), vec3(0.f, 0.f, 0.f), vec3(0.f, 0.f, 559.2f), vec3(549.6f, 0.f, 559.2f), new lambertian(white));
        d_list[i++] = new quad(vec3(343.0, 548.8, 227.0), vec3(343.0, 548.8, 332.0), vec3(213.0, 548.8, 332.0), vec3(213.0, 548.8, 227.0), new lambertian(white, 15.0f, white));
        d_list[i++] = new quad(vec3(556.0, 548.8, 0.0), vec3(556.0, 548.8, 559.2), vec3(0.0, 548.8, 559.2), vec3(0.0, 548.8, 0.0), new lambertian(white));
        d_list[i++] = new quad(vec3(549.6, 0.0, 559.2), vec3(0.0, 0.0, 559.2), vec3(0.0, 548.8, 559.2), vec3(556.0, 548.8, 559.2), new lambertian(white));
        d_list[i++] = new quad(vec3(0.0, 0.0, 559.2), vec3(0.0, 0.0, 0.0), vec3(0.0, 548.8, 0.0), vec3(0.0, 548.8, 559.2), new lambertian(green));
        d_list[i++] = new quad(vec3(552.8, 0.0, 0.0), vec3(549.6, 0.0, 559.2), vec3(556.0, 548.8, 559.2), vec3(556.0, 548.8, 0.0), new lambertian(red));
        d_list[i++] = new quad(vec3(130.0, 165.0, 65.0), vec3(82.0, 165.0, 225.0), vec3(240.0, 165.0, 272.0), vec3(290.0, 165.0, 114.0), new lambertian(white));
        d_list[i++] = new quad(vec3(290.0, 0.0, 114.0), vec3(290.0, 165.0, 114.0), vec3(240.0, 165.0, 272.0), vec3(240.0, 0.0, 272.0), new lambertian(white));
        d_list[i++] = new quad(vec3(130.0, 0.0, 65.0), vec3(130.0, 165.0, 65.0), vec3(290.0, 165.0, 114.0), vec3(290.0, 0.0, 114.0), new lambertian(white));
        d_list[i++] = new quad(vec3(82.0, 0.0, 225.0), vec3(82.0, 165.0, 225.0), vec3(130.0, 165.0, 65.0), vec3(130.0, 0.0, 65.0), new lambertian(white));
        d_list[i++] = new quad(vec3(240.0, 0.0, 272.0), vec3(240.0, 165.0, 272.0), vec3(82.0, 165.0, 225.0), vec3(82.0, 0.0, 225.0), new lambertian(white));
        d_list[i++] = new quad(vec3(423.0, 330.0, 247.0), vec3(265.0, 330.0, 296.0), vec3(314.0, 330.0, 456.0), vec3(472.0, 330.0, 406.0), new lambertian(white));
        d_list[i++] = new quad(vec3(423.0, 0.0, 247.0), vec3(423.0, 330.0, 247.0), vec3(472.0, 330.0, 406.0), vec3(472.0, 0.0, 406.0), new lambertian(white));
        d_list[i++] = new quad(vec3(472.0, 0.0, 406.0), vec3(472.0, 330.0, 406.0), vec3(314.0, 330.0, 456.0), vec3(314.0, 0.0, 456.0), new lambertian(white));
        d_list[i++] = new quad(vec3(314.0, 0.0, 456.0), vec3(314.0, 330.0, 456.0), vec3(265.0, 330.0, 296.0), vec3(265.0, 0.0, 296.0), new lambertian(white));
        d_list[i++] = new quad(vec3(265.0, 0.0, 296.0), vec3(265.0, 330.0, 296.0), vec3(423.0, 330.0, 247.0), vec3(423.0, 0.0, 247.0), new lambertian(white));


        *d_world = new hitable_list(d_list, 16);

        vec3 lookfrom(278,273,-800);
        vec3 lookat(278,273,-799);
        float dist_to_focus = (lookfrom - vec3(278, 273, 280)).length();
        float aperture = 0.025f;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 40.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera, int num_hitables) {
    for(int i=0; i < num_hitables; i++) {
        Shapes type = d_list[i]->type();
        if (type == SPHERE){
            delete ((sphere *)d_list[i])->mat_ptr;
        }
        else if (type == TRIANGLE){
            delete ((triangle *)d_list[i])->mat_ptr;
        }
        else if (type == QUAD){
            delete ((quad *)d_list[i])->mat_ptr;
        }
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main() {
    int nx = 1200;
    int ny = 1200;
    int ns = 4000;
    int tx = 32;
    int ty = 32;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable **d_list;
    int num_hitables = 16;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_cornell<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera, num_hitables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}