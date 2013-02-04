/********************************************************************************
 * Written by Emmanuel d'Angelo													*
 * Copyright 2011 EPFL/STI/IEL/LTS2.											*
 *																				*
 * This program is free software: you can redistribute it and/or modify			*
 *  it under the terms of the GNU General Public License as published by		*
 *  the Free Software Foundation, either version 3 of the License, or			*
 *  (at your option) any later version.											*
 *																				*
 *  This program is distributed in the hope that it will be useful,				*
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of				*
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the				*
 *  GNU General Public License for more details.								*
 *																				*
 *  You should have received a copy of the GNU General Public License			*
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.		*
 ********************************************************************************/
 
const int lwidth_down = 36;
const int lheight_down = 20;
__private const int lwidth_up = 40;
__private const int lheight_up = 24;
const int lwidth_geom = 34;
const int lheight_geom = 18;
__private const int lwidth_bi = 34;
__private const int lheight_bi = 18;

int mod(int a, int b)
{
	int q = a/b;
	int r = a - q*b;
	if (r<0)
		r += b;
	return r;
}

__kernel 
void G5v(
	__global const float *X,
	__global float *GX,
	__local float *sharedX,
	__constant int width, __constant int height)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int gid = y*width+x;
	int i = get_local_id(0)+2;
	int j = get_local_id(1)+2;
	int lid = j*lwidth_down+i;
	
	// Load the corresponding pixel value into shared memory
	sharedX[lid] = X[gid];
	// If on the border, also load the apron pixels
	if (j==2) {
		if (y>=2) {
			sharedX[lid-lwidth_down] = X[gid-width];
			sharedX[lid-2*lwidth_down] = X[gid-2*width];
		} else {
			if (y==1) {
				sharedX[lid-lwidth_down] = X[gid-width];
				sharedX[lid-2*lwidth_down] = X[gid];
			} else {
				sharedX[lid-lwidth_down] = X[gid+width];
				sharedX[lid-2*lwidth_down] = X[gid+2*width];
			}
		}
	}
	if (j==17) {
		if (y<height-2) {
			sharedX[lid+lwidth_down] = X[gid+width];
			sharedX[lid+2*lwidth_down] = X[gid+2*width];
		} else {
			if (y==height-2) {
				sharedX[lid+lwidth_down] = X[gid+width];
				sharedX[lid+2*lwidth_down] = X[gid];
			} else {
				sharedX[lid+lwidth_down] = X[gid-width];
				sharedX[lid+2*lwidth_down] = X[gid-2*width];
			}
		}
	}

	// Synchro
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int index=0, kidx=0;
	// This is a pre-computed Gaussian kernel
	const float g[5] = {0.01831563888873418029, 0.13533528323661269189, 1.0, 
						0.13533528323661269189, 0.01831563888873418029};
	float blurred=0.0;
	float sum = 1.30730184425069374436;
	for (index=-2; index<=2; index++, kidx++)
		blurred += g[kidx]*sharedX[lid+index*lwidth_down];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Normalize output and return
	blurred /= sum;
	GX[gid] = blurred;
}

__kernel 
void G5h(
	__global const float *X,
	__global float *GX,
	__local float *sharedX,
	__constant int width, __constant int height)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int gid = y*width+x;
	int i = get_local_id(0)+2;
	int j = get_local_id(1)+2;
	int lid = j*lwidth_down+i;
	
	// Load the corresponding pixel value into shared memory
	sharedX[lid] = X[gid];
	// If on the border, also load the apron pixels
	if (i==2) {
		if (x>=2) {
			sharedX[lid-1] = X[gid-1];
			sharedX[lid-2] = X[gid-2];
		} else {
			if (x==1) {
				sharedX[lid-1] = X[gid-1];
				sharedX[lid-2] = X[gid];
			} else {
				sharedX[lid-1] = X[gid+1];
				sharedX[lid-2] = X[gid+2];
			}
		}
	}
	if (i==33) {
		if (x<width-2) {
			sharedX[lid+1] = X[gid+1];
			sharedX[lid+2] = X[gid+2];
		} else {
			if (x==width-2) {
				sharedX[lid+1] = X[gid-1];
				sharedX[lid+2] = X[gid];
			} else {
				sharedX[lid+1] = X[gid-1];
				sharedX[lid+2] = X[gid-2];
			}
		}
	}
	// Synchro
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int index=0, kidx=0;
	const float g[5] = {0.01831563888873418029, 0.13533528323661269189, 1.0, 
						0.13533528323661269189, 0.01831563888873418029};
	float blurred=0.0;
	float sum = 1.30730184425069374436;
	for (index=-2; index<=2; index++, kidx++)
		blurred += g[kidx]*sharedX[lid+index];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Normalize output and return
	blurred /= sum;
	GX[gid] = blurred;
}

__kernel
void redux(
	__global const float *big,
	__global float *small,
	__constant int swidth, __constant int sheight)
{
	int x = get_global_id(0);	
	int y = get_global_id(1);
	int bwidth = 2*swidth;
	
	small[y*swidth+x] = big[(2*y+1)*bwidth + 2*x+1];
}

__kernel
void expand_bilinear(
	__global const float *small,
	__global float *big,
	__local float *sharedSmall,
	__constant int swidth, __constant int sheight)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int gid = y*swidth+x;
	
	int i = get_local_id(0)+1;
	int j = get_local_id(1)+1;
	int lid = j*lwidth_bi + i;
	
	// Load data to shared memory
	float I4 = small[gid];
	sharedSmall[lid] = I4;
	// Borders
	if (i==1) {
		if (x>0) sharedSmall[lid-1] = small[gid-1];
		else sharedSmall[lid-1] = I4;
	}
	if (i==32) {
		if (x<swidth-1) sharedSmall[lid+1] = small[gid+1];
		else sharedSmall[lid+1] = I4;
	}	
	if (j==1) {
		if (y>0) sharedSmall[i] = small[gid-swidth];
		else sharedSmall[i] = I4;
	}
	if (j==16) {
		if (y<sheight-1) sharedSmall[lid+lwidth_bi] = small[gid+swidth];
		else sharedSmall[lid+lwidth_bi] = I4;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Cut current pixel into four parts
	float I0, I1, I2, I3, I5, I6, I7, I8;
	// read from shared memory
	I0 = sharedSmall[lid-lwidth_bi-1];
	I1 = sharedSmall[lid-lwidth_bi];
	I2 = sharedSmall[lid-lwidth_bi+1];
	I3 = sharedSmall[lid-1];
	I5 = sharedSmall[lid+1];
	I6 = sharedSmall[lid+lwidth_bi-1];
	I7 = sharedSmall[lid+lwidth_bi];
	I8 = sharedSmall[lid+lwidth_bi+1];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	float p, q, r, s;
	
	p = 0.25*(0.25*I0 + 0.75*I1) + 0.75*(0.25*I3 + 0.75*I4);
	q = 0.25*(0.75*I1 + 0.25*I2) + 0.75*(0.75*I4 + 0.25*I5);
	r = 0.75*(0.25*I3 + 0.75*I4) + 0.25*(0.25*I6 + 0.75*I7);
	s = 0.75*(0.75*I4 + 0.25*I5) + 0.25*(0.75*I7 + 0.25*I8);
	
	p = I4;
	q = I4;
	r = I4;
	s = I4;
	
	// Write output, by doubling the values (the flow goes to the same point!)
	int bwidth = 2*swidth;
	big[2*y*bwidth+2*x] = 2.0*p;
	big[2*y*bwidth+2*x+1] = 2.0*q;
	big[(2*y+1)*bwidth+2*x] = 2.0*r;
	big[(2*y+1)*bwidth+2*x+1] = 2.0*s;
}

__kernel
void remap(
		__global const float *I,
		__global const float *Ux,
		__global const float *Uy,
		__global float *mI,
		__constant int width, constant int height)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int gid = y*width+x;

	int fx = (float)x;
	int fy = (float)y;
	
	float dx = Ux[gid] + fx;
	float dy = Uy[gid] + fy;
	
	float k, l;
	k = floor(dx);
	l = floor(dy);
	
	int ik, il;
	ik = (int)k;
	il = (int)l;
	if (dx>=width-1 || dy>=height-1 || dx<1.0 || dy<1.0)
		mI[gid] = 0.0;
	else
		mI[gid] = (l+1.0-fy)*( (k+1.0-fx)*I[il*width+ik] + (fx-k)*I[il*width+ik+1] )
				    + (fy-l)*( (k+1.0-fx)*I[(il+1)*width+ik] + (fx-k)*I[(il+1)*width+ik+1]);
}

__kernel
void gradients(
		__global const float *I,
		__global float *Dx,
		__global float *Dy,
		__local float *sharedI,
		__constant int width, __constant int height)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int gid = y*width+x;
	
	int i = get_local_id(0)+1;
	int j = get_local_id(1)+1;
	int lid = j*lwidth_geom+i;
	
	sharedI[lid] = I[gid];
	
	// Apron
	if (i==1) {
		if (x==0)
			sharedI[lid-1] = I[gid+1];
		else
			sharedI[lid-1] = I[gid-1];
	}
	if (i==32) {
		if (x==width-1)
			sharedI[lid+1] = I[gid-1];
		else
			sharedI[lid+1] = I[gid+1];
	}
	if (j==1) {
		if (y==0)
			sharedI[lid-lwidth_geom] = I[gid+width];
		else
			sharedI[lid-lwidth_geom] = I[gid-width];
	}
	if (j==16) {
		if (y==height-1)
			sharedI[lid+lwidth_geom] = I[gid-width];
		else
			sharedI[lid+lwidth_geom] = I[gid+width];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Computations
	float dx = 0.0, dy = 0.0;
	dx = 0.5*(sharedI[lid+1]-sharedI[lid-1]);
	dy = 0.5*(sharedI[lid+lwidth_geom]-sharedI[lid-lwidth_geom]);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Assign
	Dx[gid] = dx;
	Dy[gid] = dy;
}

__kernel
void initial_reg_error(
		__global const float *I0,
		__global const float *wI1,
		__global const float *Dx,
		__global const float *Dy,
		__global const float *Ux,
		__global const float *Uy,
		__global float *rho0,
		__constant int width)
{
	int gid = get_global_id(1)*width+get_global_id(0);
	rho0[gid] = wI1[gid] - I0[gid] - Dx[gid]*Ux[gid] - Dy[gid]*Uy[gid];
}

__kernel
void compute_vd(
		__global const float *Yd,
		__global float *Vd,
		__local float *sharedYd,
		__local float *sharedDYd1,
		__local float *sharedDYd2,
		__constant float mu,
		__constant float invL,
		__constant int width, 
		__constant int height)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	
	int i = get_local_id(0)+1;
	int j = get_local_id(1)+1;
	
	int lid = j*lwidth_geom + i;
	int gid = y*width + x;
	
	float yd_ij = Yd[gid];
	sharedYd[lid] = yd_ij;

// Keep it or not ?
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//---------------------- Border handling ----------------------
	if (i==1) {
		if (x!=0)
			sharedYd[lid-1] = Yd[gid-1];
		else
			sharedYd[lid-1] = 0.0;
	}
	if (i==32) {
		if (x==width-1)
			sharedYd[lid+1] = yd_ij;
		else
			sharedYd[lid+1] = Yd[gid+1];
	}
	if (j==1) {
		if (y!=0)
			sharedYd[i] = Yd[gid-width];
		else
			sharedYd[i] = 0.0;
	}
	if (j==16) {
		if (y==height-1)
			sharedYd[lid+lwidth_geom] = yd_ij;
		else
			sharedYd[lid+lwidth_geom] = Yd[gid+width];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	//------------------------------------------------------------

	// Compute derivatives, then threshold (F_mu => threshold, cf. NESTA)
	float2 grad;
	grad.x = sharedYd[lid+1]-yd_ij;
	grad.y = sharedYd[lid+lwidth_geom]-yd_ij;
	barrier(CLK_LOCAL_MEM_FENCE);

	if (i==1) {
		sharedDYd1[lid-1] = yd_ij - sharedYd[lid-1];
		sharedDYd2[lid-1] = sharedYd[lid+lwidth_geom-1] - sharedYd[lid-1];
	}
	if (j==1) {
		sharedDYd1[i] = sharedYd[i+1] - sharedYd[i];
		sharedDYd2[i] = yd_ij - sharedYd[i];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	float norm_grad = max(length(grad), mu);
		
	sharedDYd1[lid] = grad.x / norm_grad;
	sharedDYd2[lid] = grad.y / norm_grad;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Take care of the borders
	if (i==1) {
		grad.x = sharedDYd1[lid-1];
		grad.y = sharedDYd2[lid-1];
		norm_grad = max(length(grad), mu);
		sharedDYd1[lid-1] = grad.x / norm_grad;
		sharedDYd2[lid-1] = grad.y / norm_grad;
	}
	if (j==1) {
		grad.x = sharedDYd1[i];
		grad.y = sharedDYd2[i];
		norm_grad = max(length(grad), mu);
		sharedDYd1[i] = grad.x / norm_grad;
		sharedDYd2[i] = grad.y / norm_grad;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	//------------------------------------------------------------

	// Now, take the divergence
	
	float div1=0.0;
	if (x<width-1)
		div1 = sharedDYd1[lid];
	if (x>0)
		div1 = div1 - sharedDYd1[lid-1];

	float div2=0.0;
	if (y<height-1)
		div2 = sharedDYd2[lid];
	if (y>0)
		div2 = div2 - sharedDYd2[lid-lwidth_geom];
	barrier(CLK_LOCAL_MEM_FENCE);

	// The final result is minus the divergence
	Vd[gid] = yd_ij + invL*(div1+div2);
}

__kernel
void compute_updates(
		__global const float *Vk1,
		__global const float *Vk2,
		__global const float *Xkm1_1,
		__global const float *Xkm1_2,
		__global const float *Rho0,
		__global const float *DI1,
		__global const float *DI2,
		__global float *Xk1,
		__global float *Xk2,
		__global float *Ykp1_1,
		__global float *Ykp1_2,
		__constant float lambdaOverL,
		__constant float t_mix,
		__constant int width)
{
	int gid = get_global_id(1)*width+get_global_id(0);
	
	// Compute the prox here --------------------------------------------------
	float xk1_ij = Vk1[gid];
	float xk2_ij = Vk2[gid];
	
	float d1 = DI1[gid];
	float d2 = DI2[gid];
	
	float reg_error = Rho0[gid] + d1*xk1_ij + d2*xk2_ij;
	float norm_grad = d1*d1 + d2*d2;
	float threshold = lambdaOverL*norm_grad;
	if (fabs(reg_error)<=threshold) {
		if (norm_grad<0.000001) norm_grad = 1.0;
		reg_error = reg_error / norm_grad;
		xk1_ij = xk1_ij - reg_error*d1;
		xk2_ij = xk2_ij - reg_error*d2;
	} else {
		xk1_ij = xk1_ij - sign(reg_error)*lambdaOverL*d1;
		xk2_ij = xk2_ij - sign(reg_error)*lambdaOverL*d2;
	}
	
	// Compute Ykp1 here ------------------------------------------------------
	float ykp1_1 = xk1_ij + t_mix*(xk1_ij - Xkm1_1[gid]);
	float ykp1_2 = xk2_ij + t_mix*(xk2_ij - Xkm1_2[gid]);
	
	// Assign output values ---------------------------------------------------
	Xk1[gid] = xk1_ij;
	Xk2[gid] = xk2_ij;
	Ykp1_1[gid] = ykp1_1;
	Ykp1_2[gid] = ykp1_2;
}

__kernel
void copy(
		__global const float *src1,
		__global const float *src2,
		__global float *dest1,
		__global float *dest2,
		__constant int width)
{
	int gid = get_global_id(1)*width + get_global_id(0);
	dest1[gid] = src1[gid];
	dest2[gid] = src2[gid];
}


__kernel
void reset_uds(
		__global float *U1,
		__global float *U2,
		__constant int width)
{
	int gid = get_global_id(1)*width + get_global_id(0);
	U1[gid] = 0.0;
	U2[gid] = 0.0;
}
