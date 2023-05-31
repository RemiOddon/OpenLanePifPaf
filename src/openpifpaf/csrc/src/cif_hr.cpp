#include <algorithm>
#include <cmath>

#include "openpifpaf/utils.hpp"
#include "openpifpaf/decoder/utils/cif_hr.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


int64_t CifHr::neighbors = 16;
double CifHr::threshold = 0.3;
bool CifHr::ablation_skip = false;


inline float approx_exp(float x) {
    if (x > 2.0 || x < -2.0) return 0.0;
    x = 1.0 + x / 8.0;
    x *= x;
    x *= x;
    x *= x;
    return x;
}


void CifHr::accumulate(const torch::Tensor& cif_field, int64_t stride, double min_scale, double factor) {
    if (ablation_skip) return;

    auto cif_field_a = cif_field.accessor<float, 4>();
    float min_scale_f = min_scale / stride;

    float v, x, y, z, scale, sigma;//DLAV
    for (int64_t f=0; f < cif_field_a.size(0); f++) {
        for (int64_t j=0; j < cif_field_a.size(2); j++) {
            for (int64_t i=0; i < cif_field_a.size(3); i++) {
                v = cif_field_a[f][1][j][i];
                if (v < threshold) continue;//TBD

                scale = cif_field_a[f][5][j][i];//DLAV 4->5
                if (scale < min_scale_f) continue;

                x = cif_field_a[f][2][j][i];
                y = cif_field_a[f][3][j][i];
                z = cif_field_a[f][4][j][i];//DLAV
                sigma = fmaxf(1.0, 0.5 * scale);//DLAV

                // Occupancy covers 2sigma.
                // Restrict this accumulation to 1sigma so that seeds for the same joint
                // are properly suppressed.
                add_gauss(f, v / neighbors * factor, x, y, z, sigma, 1.0);//DLAV
            }
        }
    }
}


void CifHr::add_gauss(int64_t f, float v, float x, float y, float z, float sigma, float truncate) {//DLAV
    auto accumulated_a = accumulated.accessor<float, 4>();//DLAV 3->4

    auto minx = int64_t(x - truncate * sigma);//DLAV
    auto miny = int64_t(y - truncate * sigma);//DLAV
    auto minz = int64_t(z - truncate * sigma);//DLAV

    auto maxx = int64_t(x + truncate * sigma + 1);//DLAV
    auto maxy = int64_t(y + truncate * sigma + 1);//DLAV
    auto maxz = int64_t(z + truncate * sigma + 1);//DLAV

    float sigma2 = sigma * sigma ;
    float truncate2_sigma2 = truncate * truncate * sigma2;
    float deltax2, deltay2, deltaz2;//DLAV
    float vv;
    for (int64_t xx=minx; xx < maxx; xx++) {
        deltax2 = (xx - x) * (xx - x);
        for (int64_t yy=miny; yy < maxy; yy++) {
            deltay2 = (yy - y) * (yy - y);
            for (int64_t zz=minz; zz < maxz; zz++){//DLAV
                deltaz2 = (zz -z) * (zz - z);

                if (deltax2 + deltay2 + deltaz2 > truncate2_sigma2) continue;

                if (deltax2 < 0.25 && deltay2 < 0.25 && deltaz2 < 0.25) {
                    // this is the closest pixel
                    vv = v;
                } else {
                    vv = v * approx_exp(-0.5 * (deltax2 + deltay2 + deltaz2) / sigma2);
                }

                auto& entry = accumulated_a[f][xx][yy][zz];
                entry = fmaxf(entry, revision) + vv;
                entry = fminf(entry, revision + 1.0);
            }
        }
    }
}


std::tuple<torch::Tensor, double> CifHr::get_accumulated(void) {
    return { accumulated, revision };
}


void CifHr::reset(const at::IntArrayRef& shape, int64_t stride) {
    if (accumulated_buffer.size(0) < shape[0]
        || accumulated_buffer.size(1) < 150//DLAV
        || accumulated_buffer.size(2) < 40//DLAV
        || accumulated_buffer.size(3) < 20//DLAV
    ) {
        OPENPIFPAF_INFO("resizing cifhr buffer");
        accumulated_buffer = torch::zeros({
            shape[0],
            150,//DLAV
            40,//DLAV
            20//DLAV
        });
    }

    accumulated = accumulated_buffer.index({
        at::indexing::Slice(0, shape[0]),
        at::indexing::Slice(0, 150),//DLAV
        at::indexing::Slice(0, 40),//DLAV
        at::indexing::Slice(0, 20)//DLAV
    });
    revision++;

    if (revision > 10000) {
        accumulated_buffer.zero_();
        revision = 0.0;
    }
}


void CifDetHr::accumulate(const torch::Tensor& cifdet_field, int64_t stride, double min_scale, double factor) {
    auto cifdet_field_a = cifdet_field.accessor<float, 4>();
    float min_scale_f = min_scale / stride;

    float v, x, y, z, w, h, sigma;//DLAV
    for (int64_t f=0; f < cifdet_field_a.size(0); f++) {
        for (int64_t j=0; j < cifdet_field_a.size(2); j++) {
            for (int64_t i=0; i < cifdet_field_a.size(3); i++) {
                v = cifdet_field_a[f][1][j][i];
                if (v < threshold) continue;

                w = cifdet_field_a[f][5][j][i];//DLAV
                h = cifdet_field_a[f][6][j][i];//DLAV
                if (w < min_scale_f || h < min_scale_f) continue;

                x = cifdet_field_a[f][2][j][i];//DLAV
                y = cifdet_field_a[f][3][j][i];//DLAV
                z = cifdet_field_a[f][4][j][i];//DLAV
                sigma = fmaxf(1.0, 0.1 * fmin(w, h) * stride);

                // Occupancy covers 2sigma.
                // Restrict this accumulation to 1sigma so that seeds for the same joint
                // are properly suppressed.
                add_gauss(f, v / neighbors * factor, x, y, z, sigma, 1.0);
            }
        }
    }
}


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
