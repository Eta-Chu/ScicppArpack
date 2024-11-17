#include <functional>
#include <iostream>
#include <optional>
#include <tuple>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xadapt.hpp"
#include "arpack.hpp"
#include "xtensor/xtensor_forward.hpp"
#include "xtensor/xnpy.hpp"


std::tuple<xt::xarray<double>, std::optional<xt::xarray<double>>> eigsh(
        xt::xarray<double>& A, int k = 6,
        const arpack::which which = arpack::which::smallest_algebraic,
        bool return_eigvec = false
        ){

    xt::xarray<int> shape = xt::adapt(A.shape());
    arpack::which ritz_option = which;
    

    const a_int N      = shape[0];
    const a_int nev    = k;
    const a_int ncv    = 2 * nev + 1;
    const a_int ldv    = N;
    const a_int ldz    = N;
    const a_int lworkl = ncv * (ncv + 8);
    const a_int rvec   = return_eigvec; 

    const double tol = 0.0;
    const double sigma = 0.0;

    std::vector<double> resid(N);
    xt::xarray<double> V = xt::empty<double>({ncv, ldv});
    //std::vector<double> V(ldv * ncv);
    xt::xarray<double> z = xt::empty<double>({nev, ldv});
    //std::vector<double> z(ldz * nev);
    xt::xarray<double> d = xt::empty<double>({nev});
    //std::vector<double> d(nev);
    std::vector<double> workd(3 * N);
    std::vector<double> workl(lworkl);
    std::vector<a_int> select(ncv); // since HOWMNY = 'A', only used as workspace here


    a_int iparam[11], ipntr[11];
    iparam[0] = 1;      // ishift
    iparam[2] = 10 * N; // on input: maxit; on output: actual iteration
    iparam[3] = 1;      // NB, only 1 allowed
    iparam[6] = 1;      // mode
    
    a_int info = 0, ido = 0;
    do {
        arpack::saupd(ido, arpack::bmat::identity, N,
                      ritz_option, nev, tol, resid.data(), ncv,
                      V.data(), ldv, iparam, ipntr, workd.data(),
                      workl.data(), lworkl, info);
        
        xt::xarray<double> x = xt::adapt(&(workd[ipntr[0]-1]), {N, 1});
        xt::xarray<double> y = xt::linalg::dot(A, x);
        double* wy = &(workd[ipntr[1] - 1]);
        for (int i = 0; i < N; i++){
            wy[i] = y.data()[i];
        }

    } while (ido == 1 || ido == -1);

    arpack::seupd(rvec, arpack::howmny::ritz_vectors, select.data(), d.data(),
                  z.data(), ldz, sigma, arpack::bmat::identity, N,
                  ritz_option, nev, tol, resid.data(), ncv,
                  V.data(), ldv, iparam, ipntr, workd.data(),
                  workl.data(), lworkl, info);
    if (info < 0) throw std::runtime_error("Error in seupd, info " + std::to_string(info));
    
    if (return_eigvec) {
        z = xt::transpose(z);
        return {d, z};
    }
    else {
        d = xt::sort(d);
        return {d, std::nullopt};
    }
};

int main(){
    xt::xarray<double> m = xt::random::randn<double>({20, 20});
    m = xt::transpose(m) + m;
    int k = 6 ;
    arpack::which which = arpack::which::smallest_algebraic;
    int r = 0;

    auto [val, vec] = eigsh(m, k, which, r);
    std::cout << val << std::endl;
    if (vec) {
        std::cout << *vec << std::endl;
        xt::dump_npy("./double_vec.npy", *vec);
    }

    xt::dump_npy("./double_m.npy", m);

    return 1;
};

