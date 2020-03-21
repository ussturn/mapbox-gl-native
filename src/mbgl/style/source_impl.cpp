#include <mbgl/style/source_impl.hpp>

namespace mbgl {
namespace style {

Source::Impl::Impl(SourceType type_, std::string id_)
    : type(type_),
      id(std::move(id_)) {
}

void Source::Impl::setPrefetchZoomDelta(optional<uint8_t> delta) noexcept {
    prefetchZoomDelta = std::move(delta);
}

optional<uint8_t> Source::Impl::getPrefetchZoomDelta() const noexcept {
    return prefetchZoomDelta;
}

void Source::Impl::setMaxOverscaleFactor(optional<uint8_t> maxOverscaleFactor_) noexcept {
    maxOverscaleFactor = std::move(maxOverscaleFactor_);
}

optional<uint8_t> Source::Impl::getMaxOverscaleFactor() const noexcept {
    return maxOverscaleFactor;
}

} // namespace style
} // namespace mbgl
