#include <algorithm>
#include <complex>
#include <cstddef>
#include <tuple>
#include <functional>
#include <optional>
#include <vector>
#include "arpack.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xshape.hpp"
#include "xtensor/xstrided_view.hpp"

namespace linalg{

template <typename T>
class LinearOperator{

public:
    std::tuple<int, int> shape;
    std::function<T(const T&, const T&)> _matvec;
    const T& tensor;

    LinearOperator(const T& tensor, 
            std::tuple<int, int>& shape, 
            std::function<T(const T&, const T&)> matvec) : 
        tensor(tensor), shape(shape), _matvec(matvec){};

    T matvec(const T& vec){
        return _matvec(this->tensor, vec);
    // reshape function
    }; 

};

//template <typename T, typename D>
//class ArpackParams{
//
//};
//
//
//template <typename T, typename D>
//class SymmetricArpackParams{
//
//public:
//
//
//};
//
//template <typename T, typename D>
//class UnsymmetricArpackParams{
//
//public:
//    size_t n;
//    size_t k;
//    std::function<T(const T&)> matvec;
//    int mode;
//    int ncv;
//    T& v0;
//    int maxiter;
//    const which which;
//    D tol;
//
//public:
//
//    UnsymmetricArpackParams(size_t n, size_t k = 6, 
//            std::function<T(const T&)> matvec, int mode,
//            int ncv, T& v0, int maxiter, const which which, D tol) : 
//    n(n), k(k), matvec(matvec), mode(mode), ncv(ncv), v0(v0), maxiter(maxiter),
//    which(which), tol(tol){};
//
//    ~UnsymmetricArpackParams(){};
//};
//
std::tuple<std::vector<double>, std::optional<xt::xarray<double>>> eigsh(
        xt::array<double>& A, int k = 6,
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
    std::vector<double> d(nev);
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
        return std::tuple<std::vector<double>, xt::xarray<double>> res(d, z);
    }
    else {
        return std::tuple<std::vector<double>, std:std::nullopt> res(d);
    }
};

std::tuple<std::vector<double>, std::optional<xt::xarray<std::complex<double>>>> eigs(
        xt::xarray<std::complex<double>>& A, int k = 6, 
        const which which = arpack::which::largest_magnitude,
        bool eigvec = false){
        
};

}
