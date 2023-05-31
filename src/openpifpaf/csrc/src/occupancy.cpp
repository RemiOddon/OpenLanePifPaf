#include <algorithm>
#include <cmath>

#include "openpifpaf/utils.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


void Occupancy::set(int64_t f, double x, double y, double z, double sigma) {//DLAV
    if (reduction != 1.0) {
        x /= reduction;
        y /= reduction;
        z /= reduction;//DLAV
        sigma = fmax(min_scale_reduced, sigma / reduction);
    }

    auto minx = std::clamp(int64_t(x - sigma), int64_t(0), occupancy.size(1) - 1);//DLAV
    auto miny = std::clamp(int64_t(y - sigma), int64_t(0), occupancy.size(2) - 1);//DLAV
    auto minz = std::clamp(int64_t(z - sigma), int64_t(0), occupancy.size(3) - 1);//DLAV
    // # +1: for non-inclusive boundary
    // # There is __not__ another plus one for rounding up:
    // # The query in occupancy does not round to nearest integer but only
    // # rounds down.
    auto maxx = std::clamp(int64_t(x + sigma), minx + 1, occupancy.size(1));//DLAV
    auto maxy = std::clamp(int64_t(y + sigma), miny + 1, occupancy.size(2));//DLAV
    auto maxz = std::clamp(int64_t(z + sigma), minz + 1, occupancy.size(3));//DLAV
    occupancy.index_put_({f, at::indexing::Slice(minx, maxx), at::indexing::Slice(miny, maxy), at::indexing::Slice(minz, maxz)}, revision + 1);//DLAV
}


bool Occupancy::get(int64_t f, double x, double y, double z) {//DLAV
    if (f >= occupancy.size(0)) return 1;

    if (reduction != 1.0) {
        x /= reduction;
        y /= reduction;
        z /= reduction;//DLAV
    }

    auto xi = std::clamp(int64_t(x), int64_t(0), occupancy.size(1) - 1);//DLAV
    auto yi = std::clamp(int64_t(y), int64_t(0), occupancy.size(2) - 1);//DLAV
    auto zi = std::clamp(int64_t(z), int64_t(0), occupancy.size(3) - 1);//DLAV
    return occupancy.index({f, xi, yi, zi}).item<int16_t>() > revision;//DLAV
}


void Occupancy::reset(const at::IntArrayRef& shape) {
    auto i = static_cast<int64_t>(shape[1] / reduction) + 1;//DLAV
    auto j = static_cast<int64_t>(shape[2] / reduction) + 1;//DLAV
    auto k = static_cast<int64_t>(shape[3] / reduction) + 1;//DLAV

    if (occupancy_buffer.size(0) < shape[0]
        || occupancy_buffer.size(1) < i
        || occupancy_buffer.size(2) < j
        || occupancy_buffer.size(3) < k
    ) {
        OPENPIFPAF_INFO("resizing occupancy buffer");
        occupancy_buffer = torch::zeros({
            shape[0],
            std::max(i, std::max(j, k)),//DLAV
            std::max(i, std::max(j, k)),//DLAV
            std::max(i, std::max(j, k))//DLAV
        }, torch::kInt16);
    }

    occupancy = occupancy_buffer.index({
        at::indexing::Slice(0, shape[0]),
        at::indexing::Slice(0, i),//DLAV
        at::indexing::Slice(0, j),//DLAV
        at::indexing::Slice(0, k)//DLAV
    });

    clear();
}


void Occupancy::clear(void) {
    revision++;
    if (revision > 32000) {
        occupancy_buffer.zero_();
        revision = 0;
    }
}


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
