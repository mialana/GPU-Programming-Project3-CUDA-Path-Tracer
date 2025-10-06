#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include <csignal>

#include "usdUtility.h"

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    bool enableUSD = false;
#ifdef ENABLE_USD
    enableUSD = true;
#endif

    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    } else if (ext.substr(0, 4) == ".usd" && enableUSD)
    {
        loadFromUSD(filename);
        return;
    } else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        } else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        } else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        } else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation,
                                                                   newGeom.rotation,
                                                                   newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    camera.invResolution.x = cameraData["INV_RES"][0];
    camera.invResolution.y = cameraData["INV_RES"][1];

    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    // set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::createDefaultCamera()
{
    Camera& camera = this->state.camera;
    camera.resolution = glm::ivec2(800, 800);
    camera.invResolution = glm::vec2(1.0f / 800.0f, 1.0f / 800.0f);

    float fovy = 45.0f;
    state.iterations = 5000;
    state.traceDepth = 8;
    state.imageName = "VOID";

    // --- Camera orientation ---
    camera.position = glm::vec3(0.0f, 5.0f, 10.5f);
    camera.lookAt = glm::vec3(0.0f, 5.0f, 0.0f);
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f);

    // --- Derived quantities ---
    float yscaled = tan(glm::radians(fovy));
    float xscaled = yscaled * static_cast<float>(camera.resolution.x) / camera.resolution.y;
    float fovx = glm::degrees(atan(xscaled));
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2.0f * xscaled / static_cast<float>(camera.resolution.x),
                                   2.0f * yscaled / static_cast<float>(camera.resolution.y));

    // --- Allocate image buffer ---
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3(0.0f));
}

Geom buildGeomFromUsdMesh(UsdGeomMesh mesh, GfMatrix4d worldXform)
{
    // Convert to glm::mat4 (column-major)
    glm::mat4 transform = glm::make_mat4(worldXform.GetArray());

    Geom g;
    g.type = MESH;
    g.transform = transform;
    g.inverseTransform = glm::inverse(transform);
    g.invTranspose = glm::inverseTranspose(transform);
    g.materialid = 4;

    // points
    VtArray<GfVec3f> points;
    mesh.GetPointsAttr().Get(&points);

    // face count
    VtArray<int> faceVertexCounts;
    mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);

    // face indices
    VtArray<int> faceVertexIndices;
    mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);

    std::string reason;
    if (!UsdGeomMesh::ValidateTopology(faceVertexIndices, faceVertexCounts, points.size(), &reason))
    {
        std::cout << "Imported Mesh has invalid topology: " << reason << std::endl;
        raise(SIGINT);
    }

    int numPoints = points.size();
    int numTris = faceVertexCounts.size();
    int numIndices = faceVertexIndices.size();

    if (numIndices % 3 != 0 || numIndices / numTris != 3)
    {
        std::cout << "Input mesh is not properly triangulated." << std::endl;
        raise(SIGINT);
    }

    // start filling arrays
    g.mesh.vertices = new glm::vec3[numPoints];
    g.mesh.triVerts = new glm::ivec3[numTris];

    for (int i = 0; i < numPoints; i++)
    {
        // convert each GfVec3f from USD to glm::vec3.
        g.mesh.vertices[i] = glm::vec3(points[i][0], points[i][1], points[i][2]);
    }

    int i = 0;
    for (int triIdx = 0; triIdx < numTris; triIdx++)
    {
        g.mesh.triVerts[triIdx] = glm::ivec3(faceVertexIndices[i],
                                             faceVertexIndices[i + 1],
                                             faceVertexIndices[i + 2]);
        i += 3;
    }

    g.mesh.vertexCount = numPoints;
    g.mesh.triVertCount = numTris;

    return g;
}

void Scene::loadFromUSD(const std::string& usdName)
{
    UsdStageRefPtr usdStage = UsdStage::Open(usdName);
    if (!usdStage)
    {
        std::cout << "Failed to open stage:" << usdName << std::endl;
        raise(SIGINT);  // exit now
    }

    UsdGeomXformCache xformCache;

    for (const UsdPrim& prim : usdStage->Traverse())
    {
        // Check if the prim is a UsdGeomMesh
        if (prim.IsA<UsdGeomMesh>())
        {
            UsdGeomMesh mesh(prim);

            std::cout << "Found Mesh: " << mesh.GetPath() << std::endl;

            // transform
            GfMatrix4d worldXform;
            worldXform = xformCache.GetLocalToWorldTransform(prim);

            Geom baseGeom = buildGeomFromUsdMesh(mesh, worldXform);

            std::cout << "Built geom from USD mesh:" << std::endl;
            std::cout << "   Name: " << mesh.GetPath() << std::endl;
            std::cout << "   Vertices: " << baseGeom.mesh.vertexCount << std::endl;
            std::cout << "   Triangles: " << baseGeom.mesh.triVertCount << std::endl;

            geoms.push_back(baseGeom);
        }
    }

    Scene::loadFromJSON("scenes/base.json"); // import light and cornel box as scene base
}
