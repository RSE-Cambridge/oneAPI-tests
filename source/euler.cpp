#include<CL/sycl.hpp>
#include<iostream>

#define SIZE 1024

#include "cxxplot.hpp"

using namespace cl::sycl;

int main()
{
	constexpr int N = 6;
	constexpr int M = 2;

	queue myQueue(cpu_selector{});
	auto dev = myQueue.get_device();
	auto ctxt = myQueue.get_context();

	std::cout << "Selected device: " << myQueue.get_device().get_info<info::device::name>() << std::endl;

	buffer<double, 2> A(range<2>(N, N));

//	auto q = queue{}.submit([&](handler& h) {
//	  auto out =
//			  A.get_access<access::mode::write>(h);
//
//	  h.parallel_for<class ndim>(nd_range<2>(range<2>(N, N), range<2>(M, M)), [=](nd_item<2> i) {
//		id<2> ind = i.get_global_id();
//		//out[idx] = cl::sycl::sin((double)idx[0]);
//		out[ind[0]][ind[1]] = ind[0]+ind[1];
//	  });
//	});
//	q.wait();


//	double *deviceParameters = (double*) malloc_device(N*sizeof(double), decv,)

	constexpr int num_orbits = 6;
	constexpr int num_eq = 6;
	constexpr int num_params = 2;

	buffer<double, 2> y(range<2>(num_orbits, num_eq));
	buffer<double, 2> yout(range<2>(num_orbits, num_eq));
	buffer<double, 2> p(range<2>(num_orbits, num_params));

	myQueue.submit([&](handler& h) {
	  auto acc_y = y.get_access<access::mode::write>();
	  auto acc_yout = y.get_access<access::mode::write>();

	  h.parallel_for<class init_y>(range<1>(num_orbits), [=](id<1> ID) {
		for (int j = 0; j<num_eq; j++) {
			acc_y[ID[0]][j] = 0;
			acc_yout[ID[0]][j] = 0;
		}
	  });
	});

	myQueue.submit([&](handler& h) {
	  auto acc_p = p.get_access<access::mode::write>();

	  h.parallel_for<class init_params>(range<1>(num_orbits), [=](id<1> ID) {
		for (int j = 0; j<num_params; j++) {
			acc_p[ID[0]][j] = 0;
		}
	  });
	});

	auto euler = queue{}.submit([&](handler& h) {

	  double dt = 0.2;

	  auto y1 = y.get_access<access::mode::read>();
	  auto out = yout.get_access<access::mode::read_write>(h);
	  auto p1 = yout.get_access<access::mode::read>(h);

	  h.parallel_for<class ndim>(range<1>(num_orbits), [=](id<1> i) {

		for (int ieq = 0; ieq<num_eq; ieq++) {
			out[i[0]][ieq] = y1[i[0]][ieq]+out[i[0]][ieq]*dt;
		}
		//out[idx] = cl::sycl::sin((double)idx[0]);
//		out[i[0]][i[1]] = i[0]+i[1];
	  });
	});
	euler.wait();

	auto result = yout.get_access<access::mode::read>();

	std::vector<double> output;
	for (int i = 0; i<N; i++) {
		for (int j = 0; j<N; j++) {
			std::cout << result[i][j] << std::endl;
			output.push_back(result[i][j]);
		}
	}

	cxxplot::Plot<double> plot(output);
	plot.set_xlabel("x label");
	plot.set_ylabel("y label");
	plot.show_plot();

	return 0;
}