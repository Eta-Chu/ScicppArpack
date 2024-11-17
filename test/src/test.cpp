#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xadapt.hpp"
#include "arpack.hpp"
#include "xtensor/xtensor_forward.hpp"
#include "xtensor/xnpy.hpp"

template <typename T>
class LinearOperator{

public:
    std::tuple<int, int> shape;
    std::function<T(const T&, const T&)> _matvec;
    const T& mat;

    LinearOperator(std::tuple<int, int>& shape, 
            std::function<T(const T&, const T&)> matvec, const T& mat) : shape(shape), _matvec(matvec), mat(mat){};

    T matvec(T& vec){
        return _matvec(this->mat, vec);
    }; 

};

xt::xarray<double> dot(const xt::xarray<double>& mat, const xt::xarray<double>& vec){
    return xt::linalg::dot(mat, vec);
}; 

int main(){
    std::vector<size_t> x={20, 20};
    xt::xarray<double> m = xt::random::randn<double>(x);
    //xt::xarray<double> v = xt::random::randn<double>({10, 1});
    //LinearOperator<xt::xarray<double>> op(x, dot, m);
    m = xt::transpose(m) + m;


    // set nev
    int set_nev = 5;
    arpack::which ritz_option = arpack::which::smallest_algebraic;
    

    const a_int N      = x[0];
    const a_int nev    = set_nev;
    const a_int ncv    = 2 * nev + 1;
    const a_int ldv    = N;
    const a_int ldz    = N;
    const a_int lworkl = ncv * (ncv + 8);
    const a_int rvec   = 0; 

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
        xt::xarray<double> y = xt::linalg::dot(m, x);
        double* wy = &(workd[ipntr[1] - 1]);
        for (int i = 0; i < N; i++){
            wy[i] = y.data()[i];
        }

        //diagonal_matrix_vector_product(&(workd[ipntr[0] - 1]), &(workd[ipntr[1] - 1]));
    } while (ido == 1 || ido == -1);

    std::cout << "info: " << info << std::endl;
    std::cout << "iparam[4]: " << iparam[4] << std::endl;

    arpack::seupd(rvec, arpack::howmny::ritz_vectors, select.data(), d.data(),
                  z.data(), ldz, sigma, arpack::bmat::identity, N,
                  ritz_option, nev, tol, resid.data(), ncv,
                  V.data(), ldv, iparam, ipntr, workd.data(),
                  workl.data(), lworkl, info);
    if (info < 0) throw std::runtime_error("Error in seupd, info " + std::to_string(info));
    
    for (int i = 0; i < nev; ++i) {
        double val = d[i];
        std::cout << val << std::endl;
        /*eigen value order: smallest -> biggest*/
        //if (eps > tol_check) throw std::domain_error("Correct eigenvalues not computed");
    }
    std::cout << xt::adapt(d) << std::endl;
    xt::dump_npy("./mat.npy", m); 
    xt::dump_npy("./v.npy", V);
    xt::dump_npy("./z.npy", z);
    return 1;
};
