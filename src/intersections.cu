#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside)
{
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection
                     - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    } else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    } else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float meshIntersectionTest(
    Geom m, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside)
{
    // transform ray to object space
    Ray q;
    q.origin = multiplyMV(m.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(m.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_best = FLT_MAX;
    glm::vec3 tempIntersection;

    // iterate over all triangles in the mesh.
    int triCount = m.mesh.triVertCount;
    for (int i = 0; i < triCount; i++)
    {
        glm::ivec3 tri = m.mesh.triVerts[i];
        glm::vec3 v0 = m.mesh.vertices[tri.x];
        glm::vec3 v1 = m.mesh.vertices[tri.y];
        glm::vec3 v2 = m.mesh.vertices[tri.z];
        float t;
        glm::vec3 baryCoord;

        if (glm::intersectRayTriangle(q.origin, q.direction, v0, v1, v2, baryCoord))
        {
            t = baryCoord.z;
            if (t > 0.0f && t < t_best)
            {
                t_best = t;
                tempIntersection = q.origin + q.direction * t;
                glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                normal = glm::normalize(multiplyMV(m.invTranspose, glm::vec4(faceNormal, 0.0f)));
            }
        }
    }

    if (t_best < FLT_MAX)
    {
        outside = true;
        intersectionPoint = multiplyMV(m.transform, glm::vec4(tempIntersection, 1.0f));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}
