#include "setup.cuh"

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


__device__ vec3 sample(const ray& r, hitable_list **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 colour = 0.f;
    vec3 throughput = 1.f;

    for (int i = 0; i <= 50; i++){
        hit_record rec;
        if (!(*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            break;
        }
        material *mat = rec.mat_ptr;
        
        if (mat->emission_strength > 0.0f){
            if (i==0){
                colour = throughput * mat->emission_colour * mat->emission_strength;
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
                // Lambertian just at the minute
                vec3 brdf = mat->colour * 1 / M_PI;
                float solid_angle = (cos_o * emitter_area) / distance_to_light_squared;

                material *light_mat = random_emitter->mat_ptr;
                vec3 emitter_colour = light_mat->emission_colour * light_mat->emission_strength;

                // weight this sample
                colour += throughput * brdf * emitter_colour * solid_angle * cos_i;
            }
        }

        ray scattered;
        vec3 attenuation;
        // get the scattered ray
        if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
            cur_ray = scattered;
        }
        else{
            break;
        }

        throughput *= mat->colour;

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
    return colour;
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
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable_list **world, curandState *rand_state) {
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
        col += sample(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    fb[pixel_index] = sqrt(col);
}


int main() {
    int nx = 1200;
    int ny = 1200;
    int ns = 100;
    int tx = 16;
    int ty = 16;

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
    hitable_list **hit_list;
    checkCudaErrors(cudaMalloc((void **)&hit_list, sizeof(hitable_list *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_cornell<<<1,1>>>(d_list, hit_list, d_camera, nx, ny, d_rand_state2);
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
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, hit_list, d_rand_state);
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
            vec3 colour = clamp(fb[pixel_index], 0.0f, 1.0f);
            int ir = int(255.99f*colour.x);
            int ig = int(255.99f*colour.y);
            int ib = int(255.99f*colour.z);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list, hit_list, d_camera, num_hitables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(hit_list));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}