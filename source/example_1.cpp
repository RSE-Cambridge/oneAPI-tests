#include<CL/sycl.hpp>
#include<iostream>
constexpr int num = 16;

using namespace cl::sycl;

int main()
{
	auto R = range<1>{num};
	buffer<int> A{R};

	queue{}.submit([&](handler& h) {
	  auto out =
			  A.get_access<access::mode::write>(h);
	  h.parallel_for<class ex1>(R, [=](id<1> idx) {
		out[idx] = idx[0];
	  });
	});

	auto result = A.get_access<access::mode::read>();

	for (int i = 0; i<num; i++) {
		std::cout << result[i] << std::endl;
	}

	return 0;
}