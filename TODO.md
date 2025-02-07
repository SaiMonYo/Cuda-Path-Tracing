
# Basics
 - Basic brute force path tracer + lambertian BRDF
 - Add NEE
 - Metallic BRDF with a microfacet model
 - MIS NEE (BSDF samples + light samples)
 - Glossy BRDF (Specular layer + diffuse layer below the specular)

With that you'll have something that can already do decent renders. And from there, I'd say there are many different "specializations" to go through, like different themes:

# Advanced

## Light sampling:
 - Envmaps + envmaps importance sampling
 - RIS
 - ReSTIR DI
 - Light trees (aka light BVHs)

## Materials:
 - Disney BSDF
 - Energy conservation
 - Fluorescence
 - Iridescence
 - OpenPBR style BSDF
 - Layered BSDFs (i.e. proper scattering inside the layers of the BSDF)
 - Energy conservation
 - BSSRDFs

You can even look into domain specific materials such as gemstone rendering and whatnot, there are pretty cool effects to be implemented

## Path sampling:
 - ReSTIR GI/PT
 - Path guiding

## Volumetrics:
 - Single scattering homogeneous participating media
 - Multiple scattering homogeneous participating media
 - Heterogeneous participating media

## Spectral rendering:
 - Have your path tracer be fully spectral to be able to render spectrally-dependent effects properly

## More complicated integrators than a backwards path tracer:
 - Bidir path tracing
 - MLT
 - Photon mapping
 - VCM
 - ...

## Performance:
 - A proper BVH building + traversal algorithm
 - Wavefront path tracing
 - Radiance caching
 - Visibility caching techniques