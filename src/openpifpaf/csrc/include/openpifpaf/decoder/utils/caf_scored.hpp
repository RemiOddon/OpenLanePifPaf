#pragma once

#include <torch/script.h>
#include <torch/custom_class.h>

#include <tuple>
#include <vector>

#include "openpifpaf/utils.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


struct CompositeAssociation {
    float c, x1, y1, z1, x2, y2, z2, s1, s2;//DLAV

    CompositeAssociation() :
        c(0.0),
        x1(0.0), y1(0.0), z1(0.0), x2(0.0), y2(0.0), z2(0.0),
        s1(0.0), s2(0.0) { }//DLAV
    CompositeAssociation(
        float c_,
        float x1_, float y1_, float z1_, float x2_, float y2_, float z2_,
        float s1_, float s2_//DLAV
    ) :
        c(c_),
        x1(x1_), y1(y1_), z1(z1_), x2(x2_), y2(y2_), z2(z2_),
        s1(s1_), s2(s2_)//DLAV
    { }
};


struct CafScored : torch::CustomClassHolder {
    torch::TensorAccessor<float, 4UL> cifhr_a;//DLAV 3->4
    double cifhr_revision;
    double score_th;
    double cif_floor;

    std::vector<std::vector<CompositeAssociation> > forward;
    std::vector<std::vector<CompositeAssociation> > backward;

    static double default_score_th;
    static bool ablation_no_rescore;

    CafScored(
        const torch::Tensor& cifhr_,
        double cifhr_revision_,
        double score_th_ = -1.0,
        double cif_floor_ = 0.1
    ) :
        cifhr_a(cifhr_.accessor<float, 4>()),//DLAV 3->4
        cifhr_revision(cifhr_revision_),
        score_th(score_th_ >= 0.0 ? score_th_ : default_score_th),
        cif_floor(cif_floor_)
    { }
    void fill(const torch::Tensor& caf_field, int64_t stride, const torch::Tensor& skeleton);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor> > get(void);

    float cifhr_value(int64_t f, float x, float y, float z, float default_value = -1.0);
};


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
