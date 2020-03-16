#include <mbgl/util/tile_cover.hpp>
#include <mbgl/util/constants.hpp>
#include <mbgl/util/interpolate.hpp>
#include <mbgl/map/transform_state.hpp>
#include <mbgl/util/tile_cover_impl.hpp>
#include <mbgl/util/tile_coordinate.hpp>
#include <mbgl/math/log2.hpp>
#include <mbgl/util/mat3.hpp>

#include <functional>
#include <list>

namespace mbgl {

namespace {

using ScanLine = const std::function<void(int32_t x0, int32_t x1, int32_t y)>;

// Taken from polymaps src/Layer.js
// https://github.com/simplegeo/polymaps/blob/master/src/Layer.js#L333-L383
struct edge {
    double x0 = 0, y0 = 0;
    double x1 = 0, y1 = 0;
    double dx = 0, dy = 0;

    edge(Point<double> a, Point<double> b) {
        if (a.y > b.y) std::swap(a, b);
        x0 = a.x;
        y0 = a.y;
        x1 = b.x;
        y1 = b.y;
        dx = b.x - a.x;
        dy = b.y - a.y;
    }
};

// scan-line conversion
static void scanSpans(edge e0, edge e1, int32_t ymin, int32_t ymax, ScanLine scanLine) {
    double y0 = ::fmax(ymin, std::floor(e1.y0));
    double y1 = ::fmin(ymax, std::ceil(e1.y1));

    // sort edges by x-coordinate
    if ((e0.x0 == e1.x0 && e0.y0 == e1.y0) ?
        (e0.x0 + e1.dy / e0.dy * e0.dx < e1.x1) :
        (e0.x1 - e1.dy / e0.dy * e0.dx < e1.x0)) {
        std::swap(e0, e1);
    }

    // scan lines!
    double m0 = e0.dx / e0.dy;
    double m1 = e1.dx / e1.dy;
    double d0 = e0.dx > 0; // use y + 1 to compute x0
    double d1 = e1.dx < 0; // use y + 1 to compute x1
    for (int32_t y = y0; y < y1; y++) {
        double x0 = m0 * ::fmax(0, ::fmin(e0.dy, y + d0 - e0.y0)) + e0.x0;
        double x1 = m1 * ::fmax(0, ::fmin(e1.dy, y + d1 - e1.y0)) + e1.x0;
        scanLine(std::floor(x1), std::ceil(x0), y);
    }
}

// scan-line conversion
static void scanTriangle(const Point<double>& a, const Point<double>& b, const Point<double>& c, int32_t ymin, int32_t ymax, ScanLine& scanLine) {
    edge ab = edge(a, b);
    edge bc = edge(b, c);
    edge ca = edge(c, a);

    // sort edges by y-length
    if (ab.dy > bc.dy) { std::swap(ab, bc); }
    if (ab.dy > ca.dy) { std::swap(ab, ca); }
    if (bc.dy > ca.dy) { std::swap(bc, ca); }

    // scan span! scan span!
    if (ab.dy) scanSpans(ca, ab, ymin, ymax, scanLine);
    if (bc.dy) scanSpans(ca, bc, ymin, ymax, scanLine);
}

} // namespace

namespace util {

namespace {

std::vector<OverscaledTileID> tileCover(const Point<double>& tl,
                                       const Point<double>& tr,
                                       const Point<double>& br,
                                       const Point<double>& bl,
                                       const Point<double>& c,
                                       uint8_t z,
                                       uint8_t tileZoom) {
    assert(tileZoom >= z);
    const int32_t tiles = 1 << z;

    struct ID {
        int32_t x, y;
        double sqDist;
    };

    std::vector<ID> t;

    auto scanLine = [&](int32_t x0, int32_t x1, int32_t y) {
        int32_t x;
        if (y >= 0 && y <= tiles) {
            for (x = x0; x < x1; ++x) {
                const auto dx = x + 0.5 - c.x, dy = y + 0.5 - c.y;
                t.emplace_back(ID{ x, y, dx * dx + dy * dy });
            }
        }
    };

    // Divide the screen up in two triangles and scan each of them:
    // \---+
    // | \ |
    // +---\.
    scanTriangle(tl, tr, br, 0, tiles, scanLine);
    scanTriangle(br, bl, tl, 0, tiles, scanLine);

    // Sort first by distance, then by x/y.
    std::sort(t.begin(), t.end(), [](const ID& a, const ID& b) {
        return std::tie(a.sqDist, a.x, a.y) < std::tie(b.sqDist, b.x, b.y);
    });

    // Erase duplicate tile IDs (they typically occur at the common side of both triangles).
    t.erase(std::unique(t.begin(), t.end(), [](const ID& a, const ID& b) {
                return a.x == b.x && a.y == b.y;
            }), t.end());

    std::vector<OverscaledTileID> result;
    for (const auto& id : t) {
        UnwrappedTileID unwrappedId(z, id.x, id.y);
        result.emplace_back(tileZoom, unwrappedId.wrap, unwrappedId.canonical);
    }
    return result;
}

} // namespace

int32_t coveringZoomLevel(double zoom, style::SourceType type, uint16_t size) {
    zoom += util::log2(util::tileSize / size);
    if (type == style::SourceType::Raster || type == style::SourceType::Video) {
        return ::round(zoom);
    } else {
        return std::floor(zoom);
    }
}

enum class IntersectionResult : int {
    Separate,
    Intersects,
    Contains,
};

static vec3 toVec3(const vec4& v)
{
    return vec3{ v[0], v[1], v[2] };
}

static vec3 vec3Add(const vec3& a, const vec3& b)
{
    return vec3{ a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}

static vec3 vec3Sub(const vec3& a, const vec3& b)
{
    return vec3{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

static vec3 vec3Scale(const vec3& a, double s)
{
    return vec3{ a[0] * s, a[1] * s, a[2] * s };
}

static vec3 vec3Cross(const vec3& a, const vec3& b)
{
    return vec3
    {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

static double vec3Dot(const vec3& a, const vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static double vec3LengthSq(const vec3& a)
{
    return vec3Dot(a, a);
}

static double vec3Length(const vec3& a)
{
    return std::sqrt(vec3LengthSq(a));
}

static vec3 vec3Normalize(const vec3& a)
{
    return vec3Scale(a, 1.0 / vec3Length(a));
}

static double vec4Dot(const vec4& a, const vec4& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

struct AABB
{
    AABB(vec3 min_, vec3 max_)
        : min(min_)
        , max(max_)
        , center(vec3Scale(vec3Add(min_, max_), 0.5))
    { }

    vec3 closestPoint(const vec3& point) const
    {
        return
        {
            std::max(std::min(max[0], point[0]), min[0]),
            std::max(std::min(max[1], point[1]), min[1]),
            std::max(std::min(max[2], point[2]), min[2])
        };
    }

    // Computes manhattan distance to the provided point
    vec3 distanceXYZ(const vec3& point) const
    {
        vec3 vec = vec3Sub(closestPoint(point), point);

        vec[0] = std::abs(vec[0]);
        vec[1] = std::abs(vec[1]);
        vec[2] = std::abs(vec[2]);

        return vec;
    }

    AABB quadrant(int idx) const
    {
        assert(idx >= 0 && idx < 4);
        vec3 quadrantMin = min;
        vec3 quadrantMax = max;

        // This aabb is split into 4 quadrants. For each axis define in which side of the split "idx" is
        // The result for indices 0, 1, 2, 3 is: { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }
        const std::array<bool, 4> xSplit = { 0, 1, 0, 1 };
        const std::array<bool, 4> ySplit = { 0, 0, 1, 1 };

        quadrantMin[0] = xSplit[idx] ? center[0] : quadrantMin[0];
        quadrantMax[0] = xSplit[idx] ? quadrantMax[0] : center[0];

        quadrantMin[1] = ySplit[idx] ? center[1] : quadrantMin[1];
        quadrantMax[1] = ySplit[idx] ? quadrantMax[1] : center[1];

        return AABB(quadrantMin, quadrantMax);
    }

    vec3 min;
    vec3 max;
    vec3 center;
};

template <size_t N>
static Point<double> ProjectPointsToAxis(const std::array<vec3, N>& points, const vec3& origin, const vec3& axis) {
    double min = std::numeric_limits<double>::max();
    double max = -std::numeric_limits<double>::max();

    for (const vec3& point : points) {
        double projectedPoint = vec3Dot(vec3Sub(point, origin), axis);
        min = std::min(projectedPoint, min);
        max = std::max(projectedPoint, max);
    }

    return { min, max };
}

enum
{
    near_tl,
    near_tr,
    near_br,
    near_bl,
    far_tl,
    far_tr,
    far_br,
    far_bl,
};

struct AxisProjection
{
    vec3 axis;
    Point<double> projection;   // x = min, y = max
};

class Frustum
{
public:
    using PlaneArray = std::array<vec4, 6>;
    using PointArray = std::array<vec3, 8>;
    using AxisProjections = std::array<AxisProjection, 12>;

    Frustum(const PointArray& points_, const PlaneArray& planes_)
        : points(points_)
        , planes(planes_)
    {
        // Precompute a set of separating axis candidates for precise intersection tests.
        // Remaining axes not covered in basic intersection tests are: axis[] = (edges of aabb) x (edges of frustum)
        std::array<vec3, 6> frustumEdges =
        {
            vec3Sub(points[near_br], points[near_bl]),
            vec3Sub(points[near_tl], points[near_bl]),
            vec3Sub(points[far_tl], points[near_tl]),
            vec3Sub(points[far_tr], points[near_tr]),
            vec3Sub(points[far_br], points[near_br]),
            vec3Sub(points[far_bl], points[near_bl])
        };

        for (size_t i = 0; i < frustumEdges.size(); i++) {
            // Cross product [1, 0, 0] x [a, b, c] == [0, -c, b]
            // Cross product [0, 1, 0] x [a, b, c] == [c, 0, -a]
            const vec3 axis0 = { 0.0, -frustumEdges[i][2], frustumEdges[i][1] };
            const vec3 axis1 = { frustumEdges[i][2], 0.0, -frustumEdges[i][0] };

            projections[i * 2] = {
                axis0,
                ProjectPointsToAxis(points, points[0], axis0)
            };

            projections[i * 2 + 1] = {
                axis1,
                ProjectPointsToAxis(points, points[0], axis1)
            };
        }
    }

    static Frustum fromInvProjMatrix(const mat4& invProj, double worldSize, double zoom, bool flippedY = false)
    {
        // Define frustum corner points in normalized clip space
        std::array<vec4, 8> cornerCoords = 
        {
            vec4 { -1.0, 1.0, -1.0, 1.0 },
            vec4 { 1.0, 1.0, -1.0, 1.0 },
            vec4 { 1.0, -1.0, -1.0, 1.0 },
            vec4 { -1.0, -1.0, -1.0, 1.0 },
            vec4 { -1.0, 1.0, 1.0, 1.0 },
            vec4 { 1.0, 1.0, 1.0, 1.0 },
            vec4 { 1.0, -1.0, 1.0, 1.0 },
            vec4 { -1.0, -1.0, 1.0, 1.0 } 
        };

        const double scale = std::pow(2.0, zoom);

        // Transform points to tile space
        for (auto& coord : cornerCoords)
        {
            matrix::transformMat4(coord, coord, invProj);

            for (auto& component : coord)
                component *= 1.0 / coord[3] / worldSize * scale;
        }

        std::array<vec3i, 6> frustumPlanePointIndices =
        {
            vec3i { 0, 1, 2 },  // near
            vec3i { 6, 5, 4 },  // far
            vec3i { 0, 3, 7 },  // left
            vec3i { 2, 1, 5 },  // right
            vec3i { 3, 2, 6 },  // bottom
            vec3i { 0, 4, 5 }   // top
        };

        if (flippedY) {
            std::for_each(frustumPlanePointIndices.begin(), frustumPlanePointIndices.end(),
                [](vec3i& tri) { std::swap(tri[1], tri[2]); });
        }

        PlaneArray frustumPlanes;

        for (int i = 0; i < (int)frustumPlanePointIndices.size(); i++)
        {
            const vec3i indices = frustumPlanePointIndices[i];

            // Compute plane equation using 3 points on the plane
            const vec3 p0 = toVec3(cornerCoords[indices[0]]);
            const vec3 p1 = toVec3(cornerCoords[indices[1]]);
            const vec3 p2 = toVec3(cornerCoords[indices[2]]);

            const vec3 a = vec3Sub(p0, p1);
            const vec3 b = vec3Sub(p2, p1);
            const vec3 n = vec3Normalize(vec3Cross(a, b));

            frustumPlanes[i] = { n[0], n[1], n[2], -vec3Dot(n, p1) };
        }

        PointArray frustumPoints;

        for (size_t i = 0; i < cornerCoords.size(); i++)
            frustumPoints[i] = toVec3(cornerCoords[i]);

        return Frustum(std::move(frustumPoints), std::move(frustumPlanes));
    }

    IntersectionResult intersects(const AABB& aabb) const {
        // Execute separating axis test between two convex objects to find intersections
        // Each frustum plane together with 3 major axes define the separating axes
        // This implementation is conservative as it's not checking all possible axes.
        // False positive rate is ~0.5% of all cases (see intersectsPrecise).
        // Note: test only 4 points as both min and max points have zero elevation
        assert(aabb.min[2] == 0.0 && aabb.max[2] == 0.0);

        const std::array<vec4, 4> aabbPoints =
        {
            vec4 { aabb.min[0], aabb.min[1], 0.0, 1.0 },
            vec4 { aabb.max[0], aabb.min[1], 0.0, 1.0 },
            vec4 { aabb.max[0], aabb.max[1], 0.0, 1.0 },
            vec4 { aabb.min[0], aabb.max[1], 0.0, 1.0 },
        };

        bool fullyInside = true;

        for (const vec4& plane : planes)
        {
            size_t pointsInside = 0;

            for (const vec4& point : aabbPoints) {
                pointsInside += vec4Dot(plane, point) >= 0.0;
            }

            if (!pointsInside) {
                // Separating axis found, no intersection
                return IntersectionResult::Separate;
            }

            if (pointsInside != aabbPoints.size())
                fullyInside = false;
        }

        if (fullyInside) {
            return IntersectionResult::Contains;
        }

        for (int axis = 0; axis < 3; axis++) {
            double projMin = std::numeric_limits<double>::max();
            double projMax = -std::numeric_limits<double>::max();

            for (const vec3& point : points) {
                const double projectedPoint = point[axis] - aabb.min[axis];

                projMin = std::min(projMin, projectedPoint);
                projMax = std::max(projMax, projectedPoint);
            }

            if (projMax < 0 || projMin > aabb.max[axis] - aabb.min[axis]) {
                // Separating axis found, no intersection
                return IntersectionResult::Separate;
            }
        }

        return IntersectionResult::Intersects;
    }

    IntersectionResult intersectsPrecise(const AABB& aabb) const {
        IntersectionResult result = intersects(aabb);
        
        if (result == IntersectionResult::Separate)
            return result;

        const std::array<vec3, 4> aabbPoints =
        {
            vec3 { aabb.min[0], aabb.min[1], 0.0 },
            vec3 { aabb.max[0], aabb.min[1], 0.0 },
            vec3 { aabb.max[0], aabb.max[1], 0.0 },
            vec3 { aabb.min[0], aabb.max[1], 0.0 }
        };

        // For a precise SAT-test all edge cases needs to be covered
        for (const AxisProjection& proj : projections) {
            Point<double> projectedAabb = ProjectPointsToAxis(aabbPoints, points[0], proj.axis);
            const Point<double>& projectedFrustum = proj.projection;

            if (projectedFrustum.y < projectedAabb.x || projectedFrustum.x > projectedAabb.y) {
                return IntersectionResult::Separate;
            }
        }

        return IntersectionResult::Intersects;
    }

private:
    PointArray points;
    PlaneArray planes;
    AxisProjections projections;
};

std::vector<OverscaledTileID> tileCoverLod(const TransformState& state, uint8_t z, optional<uint8_t> tileZoom) {

    struct Node
    {
        AABB aabb;
        uint8_t zoom;
        uint32_t x, y;
        int16_t wrap;
        bool fullyVisible;
    };

    struct ResultTile
    {
        OverscaledTileID id;
        double sqrDist;
    };

    const double numTiles = std::pow(2.0, z);
    const double worldSize = Projection::worldSize(state.getScale());
    const uint8_t minZoom = state.getPitch() <= (60.0 / 180.0) * M_PI ? z : 0;
    const uint8_t maxZoom = z;
    const uint8_t overscaledZ = tileZoom.value_or(z);
    const bool flippedY = state.getViewportMode() == ViewportMode::FlippedY;

    auto centerPoint = TileCoordinate::fromScreenCoordinate(state, z, { state.getSize().width /2.0, state.getSize().height/2.0 }).p;

    vec3 centerCoord =
    {
        centerPoint.x,
        centerPoint.y,
        0.0
    };

    const Frustum frustum = Frustum::fromInvProjMatrix(state.getInvProjectionMatrix(), worldSize, z, flippedY);

    // There should always be a certain number of maximum zoom level tiles surrounding the center location
    const double radiusOfMaxLvlLodInTiles = 3;

    const auto newRootTile = [&](int16_t wrap) -> Node {
        return 
        {
            AABB({ wrap * numTiles, 0.0, 0.0 }, { (wrap + 1) * numTiles, numTiles, 0.0 }),
            uint8_t(0),
            uint16_t(0),
            uint16_t(0),
            wrap,
            false
        };
    };

    // Perform depth-first traversal on tile tree to find visible tiles
    std::vector<Node> stack;
    std::vector<ResultTile> result;
    stack.reserve(128);

    // World copies shall be rendered three times on both sides from closest to farthest
    for (int i = 1; i <= 3; i++) {
        stack.push_back(newRootTile(-i));
        stack.push_back(newRootTile(i));
    }

    stack.push_back(newRootTile(0));

    while (stack.size())
    {
        Node node = stack.back();
        stack.pop_back();

        // Use cached visibility information of ancestor nodes
        if (!node.fullyVisible) {
            const IntersectionResult intersection = frustum.intersects(node.aabb);

            if (intersection == IntersectionResult::Separate)
                continue;

            node.fullyVisible = intersection == IntersectionResult::Contains;
        }

        const vec3 distanceXyz = node.aabb.distanceXYZ(centerCoord);
        const double* longestDim = std::max_element(distanceXyz.data(), distanceXyz.data() + distanceXyz.size());
        assert(longestDim);

        // We're using distance based heuristics to determine if a tile should be split into quadrants or not.
        // radiusOfMaxLvlLodInTiles defines that there's always a certain number of maxLevel tiles next to the map center.
        // Using the fact that a parent node in quadtree is twice the size of its children (per dimension)
        // we can define distance thresholds for each relative level:
        // f(k) = offset + 2 + 4 + 8 + 16 + ... + 2^k. This is the same as "offset+2^(k+1)-2"
        const double distToSplit = radiusOfMaxLvlLodInTiles + (1 << (maxZoom - node.zoom)) - 2;

        // Have we reached the target depth or is the tile too far away to be any split further?
        if (node.zoom == maxZoom || (*longestDim > distToSplit && node.zoom >= minZoom))
        {
            // Perform precise intersection test between the frustum and aabb. This will cull < 1% false positives missed by the original test
            if (frustum.intersectsPrecise(node.aabb) != IntersectionResult::Separate) 
            {
                const OverscaledTileID id = { node.zoom == maxZoom ? overscaledZ : node.zoom, node.wrap, node.zoom, node.x, node.y };
                const double sqrDistance = vec3LengthSq({
                    node.wrap * numTiles + node.x + 0.5 - centerCoord[0],
                    node.y + 0.5 - centerCoord[1],
                    0.0 });

                result.push_back({ id, sqrDistance });
            }
            continue;
        }

        for (int i = 0; i < 4; i++)
        {
            const uint32_t childX = (node.x << 1) + (i % 2);
            const uint32_t childY = (node.y << 1) + (i >> 1);

            // Create child node and push to the stack for traversal
            Node child = node;

            child.aabb = node.aabb.quadrant(i);
            child.zoom = node.zoom + 1;
            child.x = childX;
            child.y = childY;

            stack.push_back(child);
        }
    }

    // Sort results by distance
    std::sort(result.begin(), result.end(), [](const ResultTile& a, const ResultTile& b) {
        return a.sqrDist < b.sqrDist;
    });

    std::vector<OverscaledTileID> ids;
    ids.reserve(result.size());

    for (const auto& tile : result) {
        ids.push_back(tile.id);
    }

    return ids;
}

std::vector<OverscaledTileID> tileCover(const LatLngBounds& bounds_, uint8_t z, optional<uint8_t> tileZoom) {
    if (bounds_.isEmpty() ||
        bounds_.south() >  util::LATITUDE_MAX ||
        bounds_.north() < -util::LATITUDE_MAX) {
        return {};
    }

    LatLngBounds bounds = LatLngBounds::hull(
        { std::max(bounds_.south(), -util::LATITUDE_MAX), bounds_.west() },
        { std::min(bounds_.north(),  util::LATITUDE_MAX), bounds_.east() });

    return tileCover(
        Projection::project(bounds.northwest(), z),
        Projection::project(bounds.northeast(), z),
        Projection::project(bounds.southeast(), z),
        Projection::project(bounds.southwest(), z),
        Projection::project(bounds.center(), z),
        z, tileZoom.value_or(z));
}

std::vector<OverscaledTileID> tileCover(const TransformState& state, uint8_t z, optional<uint8_t> tileZoom) {
    return tileCoverLod(state, z, tileZoom);
    // assert(state.valid());

    // const double w = state.getSize().width;
    // const double h = state.getSize().height;
    // return tileCover(
    //     TileCoordinate::fromScreenCoordinate(state, z, { 0,   0   }).p,
    //     TileCoordinate::fromScreenCoordinate(state, z, { w,   0   }).p,
    //     TileCoordinate::fromScreenCoordinate(state, z, { w,   h   }).p,
    //     TileCoordinate::fromScreenCoordinate(state, z, { 0,   h   }).p,
    //     TileCoordinate::fromScreenCoordinate(state, z, { w/2, h/2 }).p,
    //     z, tileZoom.value_or(z));
}

// std::vector<UnwrappedTileID> tileCover(const Geometry<double>& geometry, uint8_t z) {
//     std::vector<UnwrappedTileID> result;
//     TileCover tc(geometry, z, true);
//     while (tc.hasNext()) {
//         result.push_back(*tc.next());
//     };

//     return result;
// }

// Taken from https://github.com/mapbox/sphericalmercator#xyzbbox-zoom-tms_style-srs
// Computes the projected tiles for the lower left and upper right points of the bounds
// and uses that to compute the tile cover count
uint64_t tileCount(const LatLngBounds& bounds, uint8_t zoom){
    if (zoom == 0) {
        return 1;
    }
    auto sw = Projection::project(bounds.southwest(), zoom);
    auto ne = Projection::project(bounds.northeast(), zoom);
    auto maxTile = std::pow(2.0, zoom);
    auto x1 = floor(sw.x);
    auto x2 = ceil(ne.x) - 1;
    auto y1 = util::clamp(floor(sw.y), 0.0, maxTile - 1);
    auto y2 = util::clamp(floor(ne.y), 0.0, maxTile - 1);

    auto dx = x1 > x2 ? (maxTile - x1) + x2 : x2 - x1;
    auto dy = y1 - y2;
    return (dx + 1) * (dy + 1);
}

uint64_t tileCount(const Geometry<double>& geometry, uint8_t z) {
    uint64_t tileCount = 0;

    TileCover tc(geometry, z, true);
    while (tc.next()) {
        tileCount++;
    };
    return tileCount;
}

TileCover::TileCover(const LatLngBounds&bounds_, uint8_t z) {
    LatLngBounds bounds = LatLngBounds::hull(
        { std::max(bounds_.south(), -util::LATITUDE_MAX), bounds_.west() },
        { std::min(bounds_.north(),  util::LATITUDE_MAX), bounds_.east() });

    if (bounds.isEmpty() ||
        bounds.south() >  util::LATITUDE_MAX ||
        bounds.north() < -util::LATITUDE_MAX) {
        bounds = LatLngBounds::world();
    }

    auto sw = Projection::project(bounds.southwest(), z);
    auto ne = Projection::project(bounds.northeast(), z);
    auto se = Projection::project(bounds.southeast(), z);
    auto nw = Projection::project(bounds.northwest(), z);

    Polygon<double> p({ {sw, nw, ne, se, sw} });
    impl = std::make_unique<TileCover::Impl>(z, p, false);
}

TileCover::TileCover(const Geometry<double>& geom, uint8_t z, bool project/* = true*/)
 : impl( std::make_unique<TileCover::Impl>(z, geom, project)) {
}

TileCover::~TileCover() = default;

optional<UnwrappedTileID> TileCover::next() {
    return impl->next();
}

bool TileCover::hasNext() {
    return impl->hasNext();
}

} // namespace util
} // namespace mbgl
