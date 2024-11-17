#include <iostream>
#include "xtensor/xrandom.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "arpack.hpp"
#include "linalg.hpp"

#include "xtensor/xnpy.hpp"

int main(){
    size_t n = 100;

    xt::xarray<double> m = xt::random::randn<double>({n, n});
    m = xt::transpose(m) + m;
    int k = 6 ;
    arpack::which which = arpack::which::smallest_algebraic;
    int r = 1;
    
    auto [val, vec] = linalg::eigsh(m, k, which, r);
    
    xt::dump_npy("./debug_double_mat.npy", m);
    xt::dump_npy("./debug_double_val.npy", val);
    xt::dump_npy("./debug_double_vec.npy", *vec);

    std::vector<size_t> shape = {n, n};
    xt::xarray<double> mreal = xt::random::randn<double>(shape);
    xt::xarray<double> mimag = xt::random::randn<double>(shape);
    std::complex<double> pa(0, 1);
    xt::xarray<std::complex<double>> m2 = mreal + pa * mimag;
    m2 = xt::conj(xt::transpose(m2)) + m2;
    arpack::which which2 = arpack::which::smallest_real;

    auto [val2, vec2] = linalg::eigsh(m2, k, which2, r);

    xt::dump_npy("./debug_complex_mat.npy", m2);
    xt::dump_npy("./debug_complex_val.npy", val2);
    xt::dump_npy("./debug_complex_vec.npy", *vec2);


    return 1;
}
