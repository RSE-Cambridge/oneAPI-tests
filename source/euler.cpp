#include<CL/sycl.hpp>
#include<iostream>

constexpr int num = 16;

#define SIZE 1024

using namespace cl::sycl;

int main()
{
	queue myQueue(cpu_selector{});

	std::cout << "Selected device: " << myQueue.get_device().get_info<info::device::name>() << std::endl;

	auto R = range<1>{num};
	buffer<double> A{R};

	auto q = queue{}.submit([&](handler& h) {
	  auto out =
			  A.get_access<access::mode::write>(h);
	  h.parallel_for<class ex1>(R, [=](id<1> idx) {
		out[idx] = cl::sycl::sin((double)idx[0]);
	  });
	});
	q.wait();


	auto result = A.get_access<access::mode::read>();

	for (int i = 0; i<num; i++) {
		std::cout << result[i] << std::endl;
	}

	return 0;
}