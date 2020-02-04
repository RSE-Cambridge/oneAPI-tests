#include <CL/sycl.hpp>
#include <iostream>

#include "cxxplot.hpp"
#include "Timer.hpp"

#include "mkl_rng_sycl.hpp"
#include "mkl_sycl.hpp"
#include "mkl.h"

#define SIZE 1024

#define T1 0.15
#define T2 0.012
#define Tn 25.00

using namespace cl::sycl;

double sburstf(double x, double aval, double bval, double atval, double btval)
{
	double y;
	y = 0;

	if (x>0) {
		y = (atval*(1.00-cl::sycl::exp(-x/btval)));
	}
	else {
		y = (-(aval/bval)*x*cl::sycl::exp(x/bval));
	}

	return y;
}

int main()
{
	queue myQueue(cpu_selector{});
//	queue myQueue(gpu_selector{});
	auto dev = myQueue.get_device();
	auto ctxt = myQueue.get_context();
	std::cout << "Selected device: " << myQueue.get_device().get_info<info::device::name>() << std::endl;

	constexpr int num_orbits = 10000;
	constexpr int num_eq = 6;
	constexpr int num_params = 6;

	double* y = (double*) malloc_shared(num_orbits*num_eq*sizeof(double), dev, ctxt);
	double* yout = (double*) malloc_shared(num_orbits*num_eq*sizeof(double), dev, ctxt);
	double* params = (double*) malloc_shared(num_orbits*num_params*sizeof(double), dev, ctxt);

	for (int i = 0; i<num_orbits; i++) {
		for (int j = 0; j<num_eq; j++) {
			y[i*num_eq+j] = 0;
			yout[i*num_eq+j] = 0;
		}
		y[i*num_eq+5] = 2;
	}

	for (int i = 0; i<num_orbits; i++) {
		params[i*num_params+0] = 120;
		params[i*num_params+1] = 1.5;
		params[i*num_params+2] = 0.0045;
		params[i*num_params+3] = 0.05;
		params[i*num_params+4] = 600;
		params[i*num_params+5] = 9;
	}

	mkl::rng::philox4x32x10 engine(myQueue, 0); // basic random number generator object
	//mkl::rng::gaussian<double, mkl::rng::box_muller2> distr(0.0, 1.0); //  distribution object

	const size_t n = 10000;
	std::vector<double> r(n);

	std::vector<double> output;

	Chronos::Timer timer("timer1");
	timer.start();

	double dt = 0.00001;
	double time_steps = 1.0/0.00001;
	double t = 0;
	for (int iter = 0; iter<time_steps; iter++) {
//		std::cout << "Iter: " << iter << std::endl;
		t = t+dt;

		myQueue.submit([&](handler& h) {
		  h.parallel_for<class broomhead>(range<1>(num_orbits), [=](id<1> i) {

			// g
			yout[i[0]*num_eq+0] = y[i[0]*num_eq+1];
			// u
			yout[i[0]*num_eq+1] =
					-(((1.00/T1)+(1.00/T2))*y[i[0]*num_eq+1])-(1.00/(T1*T2)*y[i[0]*num_eq+0])
							+(1.00/(T1*T2)*y[i[0]*num_eq+2])+((1.00/T1)+(1.00/T2))*(y[i[0]*num_eq+3]-y[i[0]*num_eq+4]);
			// n
			yout[i[0]*num_eq+2] = -(y[i[0]*num_eq+2]/Tn)+(y[i[0]*num_eq+3]-y[i[0]*num_eq+4]);
			//r
			yout[i[0]*num_eq+3] = (1.00/params[i[0]*num_params+2])
					*(-y[i[0]*num_eq+3]-(params[i[0]*num_params+3]*y[i[0]*num_eq+3]*y[i[0]*num_eq+4]*y[i[0]*num_eq+4])
							+sburstf(y[i[0]*num_eq+5], params[i[0]*num_params+0], params[i[0]*num_params+1],
									params[i[0]*num_params+4], params[i[0]*num_params+5]));
			// l
			yout[i[0]*num_eq+4] = (1.00/params[i[0]*num_params+2])
					*(-y[i[0]*num_eq+4]-(params[i[0]*num_params+3]*y[i[0]*num_eq+4]*y[i[0]*num_eq+3]*y[i[0]*num_eq+3])
							+sburstf(-y[i[0]*num_eq+5], params[i[0]*num_params+0], params[i[0]*num_params+1],
									params[i[0]*num_params+4], params[i[0]*num_params+5]));
			// m
			yout[i[0]*num_eq+5] = -(y[i[0]*num_eq+3]-y[i[0]*num_eq+4]);
		  });
		});
		myQueue.wait();

		myQueue.submit([&](handler& h) {

		  h.parallel_for<class euler>(range<1>(num_orbits), [=](id<1> i) {

			for (int ieq = 0; ieq<num_eq; ieq++) {
				yout[i[0]*num_eq+ieq] = y[i[0]*num_eq+ieq]+yout[i[0]*num_eq+ieq]*dt;
			}
		  });
		});
		myQueue.wait();

//		{
//			//create buffer for random numbers
//			cl::sycl::buffer<double, 1> r_buf(r.data(), cl::sycl::range<1>{n});
//			mkl::rng::generate(distr, engine, n, r_buf); // perform generation
//		}

		myQueue.submit([&](handler& h) {
		  double dt = 0.00001;
		  h.parallel_for<class euler_copy>(range<1>(num_orbits), [=](id<1> i) {

			for (int ieq = 0; ieq<num_eq; ieq++) {
				y[i[0]*num_eq+ieq] = yout[i[0]*num_eq+ieq];
			}
		  });
		});
		myQueue.wait();

		output.push_back(0);
	}

//	for (int i = 0; i<num_orbits; i++) {
//		for (int j = 0; j<num_eq; j++) {
//			std::cout << yout[i*num_eq+j] << std::endl;
//		}
//	}

	timer.stop_timer(true);

	free(y, ctxt);
	free(yout, ctxt);
	free(params, ctxt);

	cxxplot::Plot<double> plot(output);
	plot.set_xlabel("x label");
	plot.set_ylabel("y label");
	plot.show_plot();

	return 0;
}