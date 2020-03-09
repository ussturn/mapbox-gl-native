#include <mbgl/style/expression/distance.hpp>

#include <mapbox/geojson.hpp>
#include <mapbox/geometry.hpp>
#include <mapbox/cheap_ruler.hpp>

#include <mbgl/style/conversion/json.hpp>
#include <mbgl/tile/geometry_tile_data.hpp>

#include <mbgl/util/logging.hpp>
#include <mbgl/util/string.hpp>
#include <mbgl/util/geometry_distance.hpp>

#include <rapidjson/document.h>

namespace mbgl {
namespace {
//
////Returns the distance between a point P on a segment AB.
//double distanceToLineSegment(const mapbox::geometry::point<double>& p, const mapbox::geometry::point<double>& a,const mapbox::geometry::point<double>& b) {
//    mapbox::geometry::point<double> vectorAB{(b.x - a.x), (b.y - a.y)};
//    mapbox::geometry::point<double> vectorAP{(p.x - a.x), (p.y - a.y)};
//    const auto dotProduct = [](const mapbox::geometry::point<double>& v,const mapbox::geometry::point<double>& w) {
//        return v.x * w.x + v.y * w.y;
//    };
//    
//    mapbox::cheap_ruler::CheapRuler ruler(p.y, mapbox::cheap_ruler::CheapRuler::Unit::Meters);
//    // dotProduct equals to |vectorAB| * |vectorAP| * cos();
//    const double dot1 = dotProduct(vectorAB, vectorAP);
//    if (dot1 < 0) return ruler.distance(p, a);
//
//    const double dot2 =  dotProduct(vectorAB, vectorAB);
//    if (dot1 >= dot2) return ruler.distance(p, b);
//
//    const double ratio =  dot1 / dot2;
//    const mapbox::geometry::point<double> pointOnAB = {(a.x + vectorAB.x * ratio), (a.y + vectorAB.y * ratio)};
//    return ruler.distance(p, pointOnAB);
//}
//
//double distanceToLineInMeters(const mapbox::geometry::point<double>& point, const mapbox::geometry::line_string<double>& line){
//    double ret = std::numeric_limits<double>::infinity();
//    for (size_t i = 0; i < line.size() - 1; ++i){
//        ret = std::min(ret, distanceToLineSegment(point, line[i], line[i+1]));
//    }
//    return ret;
//}
//
//double distanceToLinesInMeters(const mapbox::geometry::point<double>& point, const mapbox::geometry::multi_line_string<double>& lines){
//    double ret = std::numeric_limits<double>::infinity();
//    for (const auto& line: lines) {
//        ret = std::min(ret, distanceToLineInMeters(point, line));
//    }
//    return ret;
//}

double calculateDistance(const mbgl::GeometryTileFeature& feature,
                          const mbgl::CanonicalTileID& canonical,
                          const Feature::geometry_type& geoSet) {
    return convertGeometry(feature, canonical).match(
        [&geoSet](const mapbox::geometry::point<double>& point) -> double {
            mapbox::cheap_ruler::CheapRuler ruler(point.y, mapbox::cheap_ruler::CheapRuler::Unit::Meters);
            return geoSet.match(
             [&point, &ruler](const mapbox::geometry::point<double>& point2) {
                return ruler.distance(point, point2);
             },
            [&point, &ruler](const mapbox::geometry::multi_point<double>& points) {
                double ret = std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < points.size() - 1; ++i){
                    ret = std::min(ret, ruler.distance(point, points[i]));
                }
                return ret;
            },
             [&point, &ruler](const mapbox::geometry::line_string<double>& line) {
                   double ret = std::numeric_limits<double>::infinity();
                   for (size_t i = 0; i < line.size() - 1; ++i){
                       ret = std::min(ret, ruler.distanceToLineSegment(point, line[i], line[i+1]));
                   }
                   return ret;
            },
            [&point,&ruler](const mapbox::geometry::multi_line_string<double>& lines) {
                   double ret = std::numeric_limits<double>::infinity();
                   for (const auto& line: lines) {
                       for (size_t i = 0; i < line.size() - 1; ++i){
                         ret = std::min(ret, ruler.distanceToLineSegment(point, line[i], line[i+1]));
                       }
                   }
                   return ret;
            },
            [](const auto&) -> double { return -1.0; });
        },
        [](const auto&) -> double { return -1.0; });
}

optional<mbgl::GeoJSON> parseValue(const mbgl::style::conversion::Convertible& value_,
                                         mbgl::style::expression::ParsingContext& ctx) {
    if (isObject(value_)) {
        mbgl::style::conversion::Error error;
        auto geojson = toGeoJSON(value_, error);
        if (geojson && error.message.empty()) {
            return geojson;
        }
        ctx.error(error.message);
    }

    ctx.error("'distance' expression requires valid geojson source that contains polygon geometry type.");
    return nullopt;
}

optional<Feature::geometry_type> getGeometry(const Feature& feature, mbgl::style::expression::ParsingContext& ctx) {
    const auto type = apply_visitor(ToFeatureType(), feature.geometry);
    if (type == FeatureType::Point || type == FeatureType::LineString) {
        return feature.geometry;
    }
    ctx.error("'distance' expression requires valid geojson source that contains Point/LineString geometry type.");
    return nullopt;

}
} // namespace

namespace style {
namespace expression {

Distance::Distance(GeoJSON geojson, Feature::geometry_type geometries_)
    : Expression(Kind::Distance, type::Number),
      geoJSONSource(std::move(geojson)),
      geometries(std::move(geometries_)){}

Distance::~Distance() = default;

using namespace mbgl::style::conversion;

EvaluationResult Distance::evaluate(const EvaluationContext& params) const {
    if (!params.feature || !params.canonical) {
        return -1.0;
    }
    auto geometryType = params.feature->getType();
    // Currently only support Point/Points distance to LineString
    if (geometryType == FeatureType::Point) {
        auto distance = calculateDistance(*params.feature, *params.canonical, geometries);
        return distance;
    } 
    mbgl::Log::Warning(mbgl::Event::General,
                       "distance expression currently only support feature with Point geometry.");

    return -1.0;
}

ParseResult Distance::parse(const Convertible& value, ParsingContext& ctx) {
    if (isArray(value)) {
        // object value, quoted with ["Distance", value]
        if (arrayLength(value) != 2) {
            ctx.error("'distance' expression requires exactly one argument, but found " +
                      util::toString(arrayLength(value) - 1) + " instead.");
            return ParseResult();
        }

        auto parsedValue = parseValue(arrayMember(value, 1), ctx);
        if (!parsedValue) {
            return ParseResult();
        }

        return parsedValue->match(
            [&parsedValue, &ctx](const mapbox::geometry::geometry<double>& geometrySet) {
                if (auto ret = getGeometry(mbgl::Feature(geometrySet), ctx)) {
                    return  ParseResult(std::make_unique<Distance>(*parsedValue, std::move(*ret)));
                }
                return ParseResult();
            },
            [&parsedValue, &ctx](const mapbox::feature::feature<double>& feature) {
                if (auto ret = getGeometry(mbgl::Feature(feature), ctx)) {
                    return ParseResult(std::make_unique<Distance>(*parsedValue, std::move(*ret)));
                }
                return ParseResult();
            },
            [&parsedValue, &ctx](const mapbox::feature::feature_collection<double>& features) {
                for (const auto& feature : features) {
                    if (auto ret = getGeometry(mbgl::Feature(feature), ctx)) {
                        return ParseResult(std::make_unique<Distance>(*parsedValue, std::move(*ret)));
                    }
                }
                return ParseResult();
            },
            [&ctx](const auto&) {
                ctx.error("'distance' expression requires valid geojson that contains LineString/Point geometries.");
                return ParseResult();
            });
    }
    ctx.error("'distance' expression needs to be an array with exactly one argument.");
    return ParseResult();
}

Value convertValue(const mapbox::geojson::rapidjson_value& v) {
    if (v.IsDouble()) {
        return v.GetDouble();
    }
    if (v.IsString()) {
        return std::string(v.GetString());
    }
    if (v.IsArray()) {
        std::vector<Value> result;
        result.reserve(v.Size());
        for (const auto& m : v.GetArray()) {
            result.push_back(convertValue(m));
        }
        return result;
    }
    if (v.IsObject()) {
        std::unordered_map<std::string, Value> result;
        for (const auto& m : v.GetObject()) {
            result.emplace(m.name.GetString(), convertValue(m.value));
        }
        return result;
    }
    // Ignore other types as valid geojson only contains above types.
    return Null;
}

mbgl::Value Distance::serialize() const {
    std::unordered_map<std::string, Value> serialized;
    rapidjson::CrtAllocator allocator;
    const mapbox::geojson::rapidjson_value value = mapbox::geojson::convert(geoJSONSource, allocator);
    if (value.IsObject()) {
        for (const auto& m : value.GetObject()) {
            serialized.emplace(m.name.GetString(), convertValue(m.value));
        }
    } else {
        mbgl::Log::Error(mbgl::Event::General,
                         "Failed to serialize 'distance' expression, converted rapidJSON is not an object");
    }
    return std::vector<mbgl::Value>{{getOperator(), *fromExpressionValue<mbgl::Value>(serialized)}};
}

bool Distance::operator==(const Expression& e) const {
    if (e.getKind() == Kind::Distance) {
        auto rhs = static_cast<const Distance*>(&e);
        return geoJSONSource == rhs->geoJSONSource && geometries == rhs->geometries;
    }
    return false;
}

std::vector<optional<Value>> Distance::possibleOutputs() const {
    return { nullopt };
}

std::string Distance::getOperator() const {
    return "distance";
}

} // namespace expression
} // namespace style
} // namespace mbgl
