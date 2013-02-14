// getpixel, with neumann boundary conditions
inline float4 Tex2D(__global float4 *x, int w, int h, int i, int j)
{
	if (i < 0) i = 0;
	if (j < 0) j = 0;
	if (i >= w) i = w - 1;
	if (j >= h) j = h - 1;
	return  convert_float4(x[j*w+i]);
}


// compute x, y and t(frame) derivatives for each pixel
__kernel void ComputeDerivativesKernel(__global float4* inputImage1, __global float4* inputImage2,
                                        __global float4 *Ex,__global float4 *Ey,__global float4 *Et)
{

	uint i = get_global_id(0);
    uint j = get_global_id(1);

	uint w = get_global_size(0);
	uint h = get_global_size(1);

	int pos = j*w+i;

	Ex[pos] = (1.0/4) * ( Tex2D(inputImage1,w,h, i+1, j) - Tex2D(inputImage1,w,h, i,j)
			+ Tex2D(inputImage1,w,h, i+1, j+1) - Tex2D(inputImage1,w,h, i,j+1)
			+ Tex2D(inputImage2,w,h, i+1, j) - Tex2D(inputImage2,w,h, i,j)
			+ Tex2D(inputImage2,w,h, i+1, j+1) - Tex2D(inputImage2,w,h, i,j+1));

	Ey[pos] = (1.0/4) * ( Tex2D(inputImage1,w,h, i, j+1) - Tex2D(inputImage1,w,h, i,j)
			+ Tex2D(inputImage1,w,h, i+1, j+1) - Tex2D(inputImage1,w,h, i+1,j)
			+ Tex2D(inputImage2,w,h, i, j+1) - Tex2D(inputImage2,w,h, i,j)
			+ Tex2D(inputImage2,w,h, i+1, j+1) - Tex2D(inputImage2,w,h, i+1,j));

	Et[pos] = (1.0/4) * ( Tex2D(inputImage2,w,h, i, j) - Tex2D(inputImage1,w,h, i,j)
			+ Tex2D(inputImage2,w,h, i+1, j) - Tex2D(inputImage1,w,h, i+1,j)
			+ Tex2D(inputImage2,w,h, i, j+1) - Tex2D(inputImage1,w,h, i,j+1)
			+ Tex2D(inputImage2,w,h, i+1, j+1) - Tex2D(inputImage1,w,h, i+1,j+1));
}


// compute average u and v velocity for each pixel
__kernel void u_v_avgKernel(__global float4* u, __global float4* v,
                                        __global float4 *u_avg, __global float4 *v_avg)
{

	uint i = get_global_id(0);
    uint j = get_global_id(1);

	uint w = get_global_size(0);
	uint h = get_global_size(1);

	int pos = j*w+i;

	u_avg[pos] = (1.0/6) * (Tex2D(u,w,h, i-1, j) + Tex2D(u,w,h, i+1, j)
				+ Tex2D(u,w,h, i, j-1) + Tex2D(u,w,h, i, j+1))
			+ (1.0/12) * (Tex2D(u,w,h, i-1,j-1) + Tex2D(u,w,h, i+1,j-1)
				+ Tex2D(u,w,h, i-1,j+1) + Tex2D(u,w,h, i+1,j+1));

	v_avg[pos] = (1.0/6) * (Tex2D(v,w,h, i-1, j) + Tex2D(v,w,h, i+1, j)
				+ Tex2D(v,w,h, i, j-1) + Tex2D(v,w,h, i, j+1))
			+ (1.0/12) * (Tex2D(v,w,h, i-1,j-1) + Tex2D(v,w,h, i+1,j-1)
				+ Tex2D(v,w,h, i-1,j+1) + Tex2D(v,w,h, i+1,j+1));

	


}

// update u and v velocity for each pixel
__kernel void u_v_updateKernel(__global float4* u, __global float4* v, __global float4* u_avg, __global float4* v_avg,
                                        __global float4 *Ex,__global float4 *Ey, __global float4 *Et, const float alpha)
{

	uint i = get_global_id(0);
    uint j = get_global_id(1);

	uint w = get_global_size(0);
	uint h = get_global_size(1);

	int pos = j*w+i;


	float4 t = Ex[pos]*u_avg[pos] + Ey[pos]*v_avg[pos] + Et[pos];
	t /= alpha*alpha + Ex[pos]*Ex[pos] + Ey[pos]*Ey[pos];
	u[pos] = u_avg[pos] - Ex[pos] * t;
	


}
 