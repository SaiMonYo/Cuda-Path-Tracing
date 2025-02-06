#pragma once

#include "assert.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

constexpr float ONE_OVER_PI = 1.0f/M_PI;

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

class vec3{
public:
    union{
        struct{
            float x, y, z;
        };
        float data[3];
    };

    __host__ __device__ inline vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ inline vec3(float v) : x(v), y(v), z(v) {}
    __host__ __device__ inline vec3(const float v[3]) : x(v[0]), y(v[1]), z(v[2]) {}
    __host__ __device__ inline vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ inline vec3 operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __host__ __device__ inline vec3 operator-=(const vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    __host__ __device__ inline vec3 operator*=(const vec3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
    __host__ __device__ inline vec3 operator/=(const vec3& v) { x /= v.x; y /= v.y; z /= v.z; return *this; }

    __host__ __device__ inline vec3 operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
	__host__ __device__ inline vec3 operator/=(float scalar) { float inv_scalar = 1.0f / scalar; x *= inv_scalar; y *= inv_scalar; z *= inv_scalar; return *this; }

    __host__ __device__ inline       float& operator[](int index)       { assert(index >= 0 && index < 3); return data[index]; }
	__host__ __device__ inline const float& operator[](int index) const { assert(index >= 0 && index < 3); return data[index]; }
};

__host__ __device__ inline vec3 operator-(const vec3& v) { return vec3(-v.x, -v.y, -v.z); }

__host__ __device__ inline vec3 operator+(const vec3& left, const vec3& right) { return vec3(left.x + right.x, left.y + right.y, left.z + right.z); }
__host__ __device__ inline vec3 operator-(const vec3& left, const vec3& right) { return vec3(left.x - right.x, left.y - right.y, left.z - right.z); }
__host__ __device__ inline vec3 operator*(const vec3& left, const vec3& right) { return vec3(left.x * right.x, left.y * right.y, left.z * right.z); }
__host__ __device__ inline vec3 operator/(const vec3& left, const vec3& right) { return vec3(left.x / right.x, left.y / right.y, left.z / right.z); }

__host__ __device__ inline vec3 operator+(const vec3 & v, float scalar) {                                   return vec3(v.x + scalar,     v.y + scalar,     v.z + scalar); }
__host__ __device__ inline vec3 operator-(const vec3 & v, float scalar) {                                   return vec3(v.x - scalar,     v.y - scalar,     v.z - scalar); }
__host__ __device__ inline vec3 operator*(const vec3 & v, float scalar) {                                   return vec3(v.x * scalar,     v.y * scalar,     v.z * scalar); }
__host__ __device__ inline vec3 operator/(const vec3 & v, float scalar) { float inv_scalar = 1.0f / scalar; return vec3(v.x * inv_scalar, v.y * inv_scalar, v.z * inv_scalar); }

__host__ __device__ inline vec3 operator+(float scalar, const vec3 & v) { return vec3(scalar + v.x, scalar + v.y, scalar + v.z); }
__host__ __device__ inline vec3 operator-(float scalar, const vec3 & v) { return vec3(scalar - v.x, scalar - v.y, scalar - v.z); }
__host__ __device__ inline vec3 operator*(float scalar, const vec3 & v) { return vec3(scalar * v.x, scalar * v.y, scalar * v.z); }
__host__ __device__ inline vec3 operator/(float scalar, const vec3 & v) { return vec3(scalar / v.x, scalar / v.y, scalar / v.z); }

__host__ __device__ float dot(const vec3& left, const vec3& right){
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

__host__ __device__ vec3 cross(const vec3& left, const vec3& right){
    return vec3(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x
    );
}

__host__ __device__ float length_squared(const vec3& v){
    return dot(v, v);
}

__host__ __device__ float length(const vec3& v){
    return sqrtf(length_squared(v));
}

__host__ __device__ vec3 normalise(const vec3& v){
    return v / length(v);
}

__host__ __device__ vec3 max(const vec3& left, const vec3& right){
    return vec3(
        left.x > right.x ? left.x : right.x,
        left.y > right.y ? left.y : right.y,
        left.z > right.z ? left.z : right.z
    );
}

__host__ __device__ vec3 min(const vec3& left, const vec3& right){
    return vec3(
        left.x < right.x ? left.x : right.x,
        left.y < right.y ? left.y : right.y,
        left.z < right.z ? left.z : right.z
    );
}

__host__ __device__ float min_component(const vec3& v){
    return fminf(v.x, fminf(v.y, v.z));
}

__host__ __device__ float max_component(const vec3& v){
    return fmaxf(v.x, fmaxf(v.y, v.z));
}

__host__ __device__ vec3 clamp(const vec3& v, float low, float high){
    return max(vec3(low), min(vec3(high), v));
}

__host__ __device__ vec3 sqrt(const vec3& v){
    return vec3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

__host__ __device__ vec3 pow(const vec3& v, float p){
    return vec3(powf(v.x, p), powf(v.y, p), powf(v.z, p));
}

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) - vec3(1.0f,1.0f,0.0f);
    } while (length_squared(p) >= 1.0f);
    return p;
}

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1.0f);
    } while (length_squared(p) >= 1.0f);
    return p;
}

__device__ vec3 random_cosine_weighted_direction(curandState *local_rand_state) {
    float cos_theta_sq = curand_uniform(local_rand_state);
    float sin_theta = sqrtf(1.0f - cos_theta_sq);
    float cos_theta = sqrtf(cos_theta_sq);

    float phi = 2 * M_PI * curand_uniform(local_rand_state);

    return vec3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
}

__device__ vec3 local_to_global(const vec3& local, const vec3& normal){
    vec3 arbritrary_vec = (fabsf(normal.x) < fabsf(normal.y) && fabsf(normal.x) < fabsf(normal.z) || fabsf(normal.y) > 0.98f)
                            ? vec3(1.0f, 0.0f, 0.0f)
                            : vec3(0.0f, 1.0f, 0.0f);
    
    // create tangent and bitangent direction vectors
    vec3 u = normalise(cross(arbritrary_vec, normal));
    vec3 v = normalise(cross(normal, u));

    return local.x * u + local.y * v + local.z * normal;
}

__device__ vec3 global_to_local(const vec3& global, const vec3& normal){
    vec3 arbritrary_vec = (fabsf(normal.x) < fabsf(normal.y) && fabsf(normal.x) < fabsf(normal.z) || fabsf(normal.y) > 0.98f)
                            ? vec3(1.0f, 0.0f, 0.0f)
                            : vec3(0.0f, 1.0f, 0.0f);
    
    // create tangent and bitangent direction vectors
    vec3 u = normalise(cross(arbritrary_vec, normal));
    vec3 v = normalise(cross(normal, u));

    return vec3(dot(u, global), dot(v, global), dot(normal, global));
}

__device__ vec3 lerp(vec3 a, vec3 b, float t){
    return a * (1-t) + b * t;
}

__device__ float lerp(float a, float b, float t){
    return a * (1-t) + b * t;
}

__device__ inline float square(float a){
    return a * a;
}

__device__ inline float safe_sqrt(float a){
    return a < 0 ? 0 : sqrtf(a);
}