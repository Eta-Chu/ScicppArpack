#pragma once

#include <functional>
#include <vector>

#include "arpack.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xsort.hpp"


namespace linalg{

std::tuple<xt::xarray<double>, std::optional<xt::xarray<double>>> eigsh(
        xt::xarray<double>& A, int k = 6,
        const arpack::which which = arpack::which::smallest_algebraic,
        bool return_eigvec = false){
    
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
    xt::xarray<double> z = xt::empty<double>({nev, ldv});
    xt::xarray<double> d = xt::empty<double>({nev});
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


std::tuple<xt::xarray<double>, std::optional<xt::xarray<std::complex<double>>>> eigsh(
        xt::xarray<std::complex<double>>& A, int k = 6,
        const arpack::which which = arpack::which::smallest_real,
        bool return_eigvec = false){
    
    xt::xarray<int> shape = xt::adapt(A.shape());
    arpack::which ritz_option = which;

    const a_int N      = shape[0];
    const a_int nev    = k;
    const a_int ncv    = 2 * nev + 1;
    const a_int ldv    = N;
    const a_int ldz    = N;
    const a_int lworkl = ncv * (3 * ncv + 5);
    const a_int rvec   = return_eigvec;                   // eigenvectors omitted

    const double tol = 0.0;                // small tol => more stable checks after EV computation.
    const std::complex<double> sigma(0.0, 0.0); // not referenced in this mode

    std::vector<std::complex<double>> resid(N);
    xt::xarray<std::complex<double>> V = xt::empty<std::complex<double>>({ncv, ldv});
    xt::xarray<std::complex<double>> z = xt::empty<std::complex<double>>({nev, ldz});
    xt::xarray<std::complex<double>> d = xt::empty<std::complex<double>>({nev});
    std::vector<std::complex<double>> workd(3 * N);
    std::vector<std::complex<double>> workl(lworkl);
    std::vector<std::complex<double>> workev(2 * ncv);
    std::vector<double> rwork(ncv);
    std::vector<a_int> select(ncv); // since HOWMNY = 'A', only used as workspace here

    a_int iparam[11], ipntr[14];
    iparam[0] = 1;       // ishift
    iparam[2] = 10 * N;  // on input: maxit; on output: actual iteration
    iparam[3] = 1;       // NB, only 1 allowed
    iparam[6] = 1;       // mode

    a_int info = 0, ido = 0;
    do {
        arpack::naupd(ido, arpack::bmat::identity, N,
                      ritz_option, nev, tol, resid.data(), ncv,
                      V.data(), ldv, iparam, ipntr, workd.data(),
                      workl.data(), lworkl, rwork.data(), info);
    
        xt::xarray<std::complex<double>> x = xt::adapt(&(workd[ipntr[0]-1]), {N, 1});
        xt::xarray<std::complex<double>> y = xt::linalg::dot(A, x);
        std::complex<double>* wy = &(workd[ipntr[1] - 1]);
        for (int i = 0; i < N; i++){
            wy[i] = y.data()[i];
        }   
    } while (ido == 1 || ido == -1);
    
    // check info and number of ev found by arpack.
    if (info < 0 || iparam[4] < nev) { /*arpack may succeed to compute more EV than expected*/
        std::cout << "ERROR in naupd: iparam[4] " << iparam[4] << ", nev " << nev
                << ", info " << info << std::endl;
        throw std::domain_error("Error inside ARPACK routines");
    }
    
    arpack::neupd(rvec, arpack::howmny::ritz_vectors, select.data(), d.data(),
                  z.data(), ldz, sigma, workev.data(), arpack::bmat::identity, N,
                  ritz_option, nev, tol, resid.data(), ncv,
                  V.data(), ldv, iparam, ipntr, workd.data(),
                  workl.data(), lworkl, rwork.data(), info);
    if (info < 0) throw std::runtime_error("Error in neupd, info " + std::to_string(info));
    
    xt::xarray<double> dreal = xt::real(d);
    if (return_eigvec) {
        z = xt::transpose(z);
        return {dreal, z};
    }
    else {
        dreal = xt::sort(dreal);
        return {dreal, std::nullopt};
    }
};

template <typename T>
class LinearOperator{

public:
    std::vector<size_t> shape;
    std::function<T(T&, T&)> _matvec;

    LinearOperator(std::vector<size_t>& shape, 
            std::function<T(T&, T&)> matvec) : shape(shape), _matvec(matvec){};

    T matvec(T& mat, T& vec){
        return _matvec(mat, vec);
    }; 

};
}

