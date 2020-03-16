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

struct AABB
{
    AABB(vec3 min_, vec3 max_)
        : min(min_)
        , max(max_)
        , center(vec3Scale(vec3Add(min_, max_), 0.5f))
    { }


    vec3 min;
    vec3 max;
    vec3 center;
};

class Frustum
{
public:
    using PlaneArray = std::array<vec4, 6>;
    using PointArray = std::array<vec4, 8>;

    Frustum(const PointArray& points_, const PlaneArray& planes_)
        : points(points_)
        , planes(planes_)
    { }

    static Frustum fromInvProjMatrix(const mat4& invProj, double worldSize, double zoom)
    {
        // Define frustum corner points in normalized clip space
        PointArray frustumCoords = 
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
        for (auto& coord : frustumCoords)
        {
            matrix::transformMat4(coord, coord, invProj);

            for (auto& component : coord)
                component *= 1.0 / coord[3] / worldSize * scale;
        }

        const std::array<vec3i, 6> frustumPlanePointIndices =
        {
            vec3i { 0, 1, 2 },  // near
            vec3i { 6, 5, 4 },  // far
            vec3i { 0, 3, 7 },  // left
            vec3i { 2, 1, 5 },  // right
            vec3i { 3, 2, 6 },  // bottom
            vec3i { 0, 4, 5 }   // top
        };

        PlaneArray frustumPlanes;

        for (int i = 0; i < (int)frustumPlanePointIndices.size(); i++)
        {
            const vec3i indices = frustumPlanePointIndices[i];

            // Compute plane equation using 3 points on the plane
            const vec3 p0 = toVec3(frustumCoords[indices[0]]);
            const vec3 p1 = toVec3(frustumCoords[indices[1]]);
            const vec3 p2 = toVec3(frustumCoords[indices[2]]);

            const vec3 a = vec3Sub(p0, p1);
            const vec3 b = vec3Sub(p2, p1);
            const vec3 n = vec3Normalize(vec3Cross(a, b));

            frustumPlanes[i] = { n[0], n[1], n[2], -vec3Dot(n, p1) };
        }

        return Frustum(std::move(frustumCoords), std::move(frustumPlanes));
    }

private:
    PointArray points;
    PlaneArray planes;
};

std::vector<OverscaledTileID> tileCoverLod(const TransformState& state, uint8_t z, optional<uint8_t> tileZoom) {

    struct Node
    {
        AABB aabb;
        uint8_t zoom;
        double x, y;
        int wrap;
        bool fullyVisible;
    };

    // Get center point of the map in normalized world coordinates
    vec3 centerPoint =
    { 
        state.getX() / util::tileSize,
        state.getY() / util::tileSize,
        0.0
    };

    const double numTiles = std::pow(2.0, z);
    const double worldSize = Projection::worldSize(state.getScale());
    const double minZoom = 0;
    const double maxZoom = z;

    // There should always be a certain number of maximum zoom level tiles surrounding the center location
    const double radiusOfMaxLvlLodInTiles = 3;

    const auto newRootTile = [&](int wrap) -> Node {
        return 
        {
            AABB({ wrap * numTiles, 0.0, 0.0 }, { (wrap + 1) * numTiles, numTiles, 0.0 }),
            z,
            0.0,
            0.0,
            wrap,
            false
        };
    };

    // Perform depth-first traversal on tile tree to find visible tiles
    std::vector<Node> stack;
    std::vector<OverscaledTileID> result;

    (void)centerPoint;
    (void)state;
    (void)worldSize;
    (void)minZoom;
    (void)maxZoom;
    (void)radiusOfMaxLvlLodInTiles;
    (void)newRootTile;
    (void)tileZoom;

    //state.getProjectionMatrix

    return { };
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
    assert(state.valid());

    const double w = state.getSize().width;
    const double h = state.getSize().height;
    return tileCover(
        TileCoordinate::fromScreenCoordinate(state, z, { 0,   0   }).p,
        TileCoordinate::fromScreenCoordinate(state, z, { w,   0   }).p,
        TileCoordinate::fromScreenCoordinate(state, z, { w,   h   }).p,
        TileCoordinate::fromScreenCoordinate(state, z, { 0,   h   }).p,
        TileCoordinate::fromScreenCoordinate(state, z, { w/2, h/2 }).p,
        z, tileZoom.value_or(z));
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
