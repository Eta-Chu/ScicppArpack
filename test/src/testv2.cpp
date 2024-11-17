#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xadapt.hpp"
#include "arpack.hpp"
#include "xtensor/xshape.hpp"
#include "xtensor/xtensor_forward.hpp"
#include "xtensor/xnpy.hpp"
#include "xtl/xvariant_impl.hpp"


int main(){
    std::vector<size_t> shape = {20, 20};
    xt::xarray<double> mreal = xt::random::randn<double>(shape);
    xt::xarray<double> mimag = xt::random::randn<double>(shape);
    std::complex<double> pa(0, 1);
    xt::xarray<std::complex<double>> m = mreal + pa * mimag;
    m = xt::conj(xt::transpose(m)) + m;
    xt::dump_npy("./complex_m.npy", m);
    
    int k = 5;
    bool return_eigvec = true;
    arpack::which ritz_option = arpack::which::smallest_real;

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
    //std::vector<std::complex<double>> V(ldv * ncv);
    xt::xarray<std::complex<double>> z = xt::empty<std::complex<double>>({nev, ldz});
    //std::vector<std::complex<double>> z(ldz * nev);
    xt::xarray<std::complex<double>> d = xt::empty<std::complex<double>>({nev});
    //std::vector<std::complex<double>> d(nev);
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
        xt::xarray<std::complex<double>> y = xt::linalg::dot(m, x);
        std::complex<double>* wy = &(workd[ipntr[1] - 1]);
        for (int i = 0; i < N; i++){
            wy[i] = y.data()[i];
        }   

    //diagonal_matrix_vector_product(&(workd[ipntr[0] - 1]), &(workd[ipntr[1] - 1]));
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
    
    std::cout << xt::real(d) << std::endl;
    std::cout << z << std::endl;
    std::cout << V << std::endl;
    return 1;
};
