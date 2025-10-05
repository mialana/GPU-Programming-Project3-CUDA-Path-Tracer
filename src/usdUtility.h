#pragma once

#ifdef ENABLE_USD

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#pragma GCC diagnostic ignored "-Wdeprecated"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>

#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/xformCache.h>

#include <pxr/base/vt/array.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3d.h>

#pragma GCC diagnostic pop

PXR_NAMESPACE_USING_DIRECTIVE

#endif
