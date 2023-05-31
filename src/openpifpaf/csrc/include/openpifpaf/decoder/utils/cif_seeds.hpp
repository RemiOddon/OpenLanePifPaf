#pragma once

#include <torch/script.h>
#include <torch/custom_class.h>

#include <tuple>
#include <vector>

#include "openpifpaf/utils.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


struct Seed {
    int64_t f;
    float v, x, y, z, s;//DLAV

    Seed(int64_t f_, float v_, float x_, float y_, float z_, float s_)
        : f(f_), v(v_), x(x_), y(y_), z(z_), s(s_) { }
};


struct DetSeed {
    int64_t c;
    float v, x, y, z, w, h;//DLAV

    DetSeed(int64_t c_, float v_, float x_, float y_, float z_, float w_, float h_)//DLAV
        : c(c_), v(v_), x(x_), y(y_), z(z_), w(w_), h(h_) { }//DLAV
};


struct CifSeeds : torch::CustomClassHolder {
    torch::TensorAccessor<float, 4UL> cifhr_a;//DLAV 3->4
    double cifhr_revision;
    std::vector<Seed> seeds;

    static double threshold;
    static bool ablation_nms;
    static bool ablation_no_rescore;

    CifSeeds(const torch::Tensor& cifhr_, double cifhr_revision_)
    : cifhr_a(cifhr_.accessor<float, 4>()),//DLAV 3->4
      cifhr_revision(cifhr_revision_)
    { }
    void fill(const torch::Tensor& cif_field, int64_t stride);
    std::tuple<torch::Tensor, torch::Tensor> get(void);
};


struct CifDetSeeds : torch::CustomClassHolder {
    torch::TensorAccessor<float, 4UL> cifhr_a;//DLAV
    double cifhr_revision;
    std::vector<DetSeed> seeds;

    static double threshold;

    CifDetSeeds(const torch::Tensor& cifhr_, double cifhr_revision_)
    : cifhr_a(cifhr_.accessor<float, 4>()),//DLAV
      cifhr_revision(cifhr_revision_)
    { }
    void fill(const torch::Tensor& cifdet_field, int64_t stride);
    std::tuple<torch::Tensor, torch::Tensor> get(void);
};


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
