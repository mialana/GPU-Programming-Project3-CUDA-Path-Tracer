#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include <stream_compaction/shared.h>
#include <stream_compaction/radix.h>

#define ERRORCHECK 1

#ifndef checkCUDAError
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif  // _WIN32
    exit(EXIT_FAILURE);
#endif  // ERRORCHECK
}
#endif

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int iter,
                                                                         int index,
                                                                         int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(
    uchar4* pbo, glm::ivec2 resolution, glm::vec2 invResolution, int iter, glm::vec3* image)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int y = (int)(tid * invResolution.x);
    int x = tid - y * resolution.x;

    if (x < resolution.x && y < resolution.y)
    {
        glm::vec3 pix = image[tid];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[tid].w = 0;
        pbo[tid].x = color.x;
        pbo[tid].y = color.y;
        pbo[tid].z = color.z;
    }
}

__global__ void kernCreateIndexBuffer(int n, int* dev_buf)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= n)
    {
        return;
    }

    dev_buf[index] = index;
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;

static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// the ping-pong device buffers that we will use for radix sort and stream compaction
static int* dev_pathIndicesA = NULL;
static int* dev_pathIndicesB = NULL;

static int* dev_arrA = NULL;  // dev_bools in compact, then dev_valuesA in sort
static int* dev_arrB = NULL;  // dev_indices in compact, then dev_valuesB in sort

static int* dev_blockSums = NULL;

static int* dev_materialIDs = NULL;

int* partitions = NULL;

// 1D block for path tracing
const int blockSize1d = 128;
const int maxBitLengthMaterialID = 3;

template<typename T>
__global__ void scatterFromIndices(const int n, const int* indices, const T* in, T* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    int src = indices[i];
    out[i] = in[src];
}

// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms,
               scene->geoms.data(),
               scene->geoms.size() * sizeof(Geom),
               cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials,
               scene->materials.data(),
               scene->materials.size() * sizeof(Material),
               cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    cudaMalloc(&dev_pathIndicesA, pixelcount * sizeof(int));
    cudaMalloc(&dev_pathIndicesB, pixelcount * sizeof(int));

    cudaMalloc(&dev_arrA, pixelcount * sizeof(int));
    cudaMalloc(&dev_arrB, pixelcount * sizeof(int));

    unsigned long long paddedN = 1 << ilog2ceil(pixelcount);
    int totalBlocks = divup(paddedN, 2 * blockSize1d);

    cudaMalloc(&dev_blockSums, totalBlocks * sizeof(int));

    cudaMalloc(&dev_materialIDs, pixelcount * sizeof(int));

    int num_materials = hst_scene->materials.size();
    partitions = new int[num_materials];

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    cudaFree(dev_pathIndicesA);
    cudaFree(dev_pathIndicesB);
    cudaFree(dev_arrA);
    cudaFree(dev_arrB);
    cudaFree(dev_blockSums);
    cudaFree(dev_materialIDs);

    checkCUDAError("pathtraceFree");
}

void pathtraceResetImage(int pixelcount)
{
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    checkCUDAError("pathtraceResetImage");
}

/**
 * Generate PathSegments with rays from the camera through the screen into the
 * scene, which is the first bounce of rays.
 *
 * Antialiasing - add rays for sub-pixel sampling
 * motion blur - jitter rays "in time"
 * lens effect - jitter ray origin positions based on a lens
 */
__global__ void generateRayFromCamera(Camera cam,
                                      int iter,
                                      int traceDepth,
                                      PathSegment* pathSegments)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int y = (int)(tid * cam.invResolution.x);
    int x = tid - y * cam.resolution.x;

    if (x < cam.resolution.x && y < cam.resolution.y)
    {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, tid, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        PathSegment& segment = pathSegments[tid];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // currently, ray only lands at origin of each pixel.
        // jittering within each pixel effectively creates an anti-aliasing effect.
        const float jitterAmt_x = u01(rng) * cam.pixelLength.x;
        const float jitterAmt_y = u01(rng) * cam.pixelLength.y;

        segment.ray.direction = glm::normalize(
            cam.view
            - cam.right * cam.pixelLength.x
                  * (((float)x + jitterAmt_x) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y
                  * (((float)y + jitterAmt_y) - (float)cam.resolution.y * 0.5f));

        segment.pixelIndex = tid;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int depth,
                                     int num_paths,
                                     PathSegment* pathSegments,
                                     Geom* geoms,
                                     int geoms_size,
                                     ShadeableIntersection* intersections,
                                     const int* pathIndices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_paths)
    {
        int pathIndex = pathIndices[index];
        PathSegment pathSegment = pathSegments[pathIndex];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            } else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            } else
            {
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[pathIndex].t = -1.0f;
            intersections[pathIndex].materialId = (1 << maxBitLengthMaterialID) - 1;
        } else
        {
            // The ray hits something
            intersections[pathIndex].t = t_min;
            intersections[pathIndex].materialId = geoms[hit_geom_index].materialid;
            intersections[pathIndex].surfaceNormal = normal;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(int iter,
                                  int startIdx,
                                  int endIdx,
                                  ShadeableIntersection* shadeableIntersections,
                                  PathSegment* pathSegments,
                                  Material* materials,
                                  const int* pathIndices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < endIdx && idx >= startIdx)
    {
        int pathIndex = pathIndices[idx];
        ShadeableIntersection intersection = shadeableIntersections[pathIndex];
        if (intersection.t > 0.0f)  // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f)
            {
                pathSegments[pathIndex].color *= (materialColor * material.emittance);
                pathSegments[pathIndex].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else
            {
                // use `interactions::scatterRay` to calculate bsdf value
                // offset ray by EPSILON
                scatterRay(pathSegments[pathIndex],
                           (pathSegments[pathIndex].ray.origin
                            + (intersection.t * pathSegments[pathIndex].ray.direction))
                               + (intersection.surfaceNormal * EPSILON),
                           intersection.surfaceNormal,
                           material,
                           rng);
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        } else
        {
            pathSegments[pathIndex].color = glm::vec3(0.0f);
            pathSegments[pathIndex].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

__global__ void createActivePathsBools(int n,
                                       const int* pathIndices,
                                       const PathSegment* pathSegments,
                                       int* bools)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int segIndex = pathIndices[i];
        bools[i] = (pathSegments[segIndex].remainingBounces > 0) ? 1 : 0;
    }
}

__global__ void extractMaterialIDs(int n,
                                   const int* pathIndices,
                                   const ShadeableIntersection* intersections,
                                   int* materialIDs)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= n)
    {
        return;
    }

    int src = pathIndices[index];
    materialIDs[index] = intersections[src].materialId;
}

// partition materialIDs based on where they end
__global__ void partitionMaterialIDs(int n,
                                     int* partitions,
                                     const int* pathIndices,
                                     const int* materialIDs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n - 1)
    {
        return;
    }

    int matID = materialIDs[index];
    int next_matID = materialIDs[index + 1];

    if (matID < next_matID)
    {
        partitions[matID] = index;
    }
    if (index == n - 2)
    {
        partitions[next_matID] = n - 1;  // add index for very last partition
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

    int num_materials = hst_scene->materials.size();

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<numBlocksPixels, blockSize1d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    kernCreateIndexBuffer<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_pathIndicesA);

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(depth,
                                                                           num_paths,
                                                                           dev_paths,
                                                                           dev_geoms,
                                                                           hst_scene->geoms.size(),
                                                                           dev_intersections,
                                                                           dev_pathIndicesA);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        if (guiData->SortByMaterial && ((depth & guiData->SortingPeriod) == 0)
            && num_paths > guiData->SortByMaterial)
        {
            extractMaterialIDs<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths,
                                                                             dev_pathIndicesA,
                                                                             dev_intersections,
                                                                             dev_materialIDs);

            StreamCompaction::Radix::sortByKey(num_paths,
                                               dev_materialIDs,
                                               dev_arrA,
                                               dev_pathIndicesA,
                                               dev_pathIndicesB,
                                               dev_blockSums,
                                               dev_arrB,
                                               maxBitLengthMaterialID,
                                               blockSize1d);
            checkCUDAErrorFn("sort by key");

            std::swap(dev_pathIndicesA, dev_pathIndicesB);
            std::swap(dev_materialIDs, dev_arrA);

            // use dev_arrA to store partitions as it is currently not needed for anything
            partitionMaterialIDs<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths,
                                                                               dev_arrA,
                                                                               dev_pathIndicesA,
                                                                               dev_materialIDs);

            cudaMemcpy(partitions, dev_arrA, sizeof(int) * num_materials, cudaMemcpyDeviceToHost);

            int startIdx = 0;
            int endIdx;
            for (int id = 0; id < num_materials; id++)
            {
                endIdx = partitions[id];

                shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(iter,
                                                                                startIdx,
                                                                                endIdx,
                                                                                dev_intersections,
                                                                                dev_paths,
                                                                                dev_materials,
                                                                                dev_pathIndicesA);
                startIdx = endIdx;
            }
        } else
        {
            shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(iter,
                                                                            0,
                                                                            num_paths,
                                                                            dev_intersections,
                                                                            dev_paths,
                                                                            dev_materials,
                                                                            dev_pathIndicesA);
        }

        cudaDeviceSynchronize();

        createActivePathsBools<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths,
                                                                             dev_pathIndicesA,
                                                                             dev_paths,
                                                                             dev_arrA);

        // run compact with already generated bools arr, which indicate if path is still active
        num_paths = StreamCompaction::Shared::compactWithBools(num_paths,         // n
                                                               dev_pathIndicesA,  // idata
                                                               dev_pathIndicesB,  // odata
                                                               dev_arrA,          // bools
                                                               dev_arrB,          // indices
                                                               dev_blockSums,
                                                               blockSize1d);
        checkCUDAErrorFn("compact with bools");

        // ping-pong
        std::swap(dev_pathIndicesA, dev_pathIndicesB);

        if (depth >= traceDepth || num_paths <= 0)
        {
            iterationComplete = true;  // TODO: should be based off stream compaction results.
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    finalGather<<<numBlocksPixels, blockSize1d>>>(dev_path_end - dev_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<numBlocksPixels, blockSize1d>>>(pbo,
                                                     cam.resolution,
                                                     cam.invResolution,
                                                     iter,
                                                     dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(),
               dev_image,
               pixelcount * sizeof(glm::vec3),
               cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
