 /*
 * Ka¿dy w¹tek oblicza piksel, poprzez zastosowanie filtrów
 * w grupie 8 s¹siednich pikseli w obu kierunkach X i Y. 
 * Oba filtry s¹ sumowane (suma wektorowa), tworz¹c koñcowy wynik.
 */

__kernel void gauss_filter(__global uchar4* inputImage, __global uchar4* outputImage)
{
	uint x = get_global_id(0);
    uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);

	float4 G = (float4)(0);
	float4 sigma = 0.0625f;
	
	int c = x + y * width;

	if( x >= 1 && x < (width-1) && y >= 1 && y < (height - 1))
	{
		float4 i00 = convert_float4(inputImage[c - 1 - width]);
		float4 i10 = convert_float4(inputImage[c - width]);
		float4 i20 = convert_float4(inputImage[c + 1 - width]);
		float4 i01 = convert_float4(inputImage[c - 1]);
		float4 i11 = convert_float4(inputImage[c]);
		float4 i21 = convert_float4(inputImage[c + 1]);
		float4 i02 = convert_float4(inputImage[c - 1 + width]);
		float4 i12 = convert_float4(inputImage[c + width]);
		float4 i22 = convert_float4(inputImage[c + 1 + width]);

		// Gauss
		/*
		{1 2 1}
		{2 4 2}
		{1 2 1}
		*/
		G = sigma*(float4)(1) * i00 + sigma*(float4)(1) * i10 + sigma*(float4)(1) * i20 + sigma*(float4)(1) * i01 + sigma*(float4)(2) * i11 + sigma*(float4)(1) * i21 + sigma*(float4)(1) * i02  + sigma*(float4)(1) * i12 + sigma*(float4)(1) * i22;
		// Obraz wyjœciowy
		//outputImage[c] = convert_uchar4(G);
		outputImage[c] = inputImage[c];
	}
}

int gradToAngle(float x, float y)
{
	//K¹t
	float dir = atan2(y, x)/(float)(3.14156f) * (float)(180.0f);
	//Kierunek
	if (((dir > 22.5f) && (dir < 67.5f)) || ((dir < -112.5f) && (dir > -157.5f)))
		return 45;
	else if (((dir > 67.5f) && (dir < 112.5f)) || ((dir < -67.5f) && (dir > -112.5f)))
		return 90;
	else if (((dir > 112.5f) && (dir < 157.5f)) || ((dir < -22.5f) && (dir > -67.5f)))
		return 135;
	else if (((dir < 22.5f) && (dir > -22.5f)) || ((dir > -157.5f) && (dir < -157.5f)))
		return 0;
}

__kernel void sobel_filter(__global uchar4* inputImage, __global uchar4* outputImage, __global int* outputDir)
{
	uint x = get_global_id(0);
    uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);

	float4 Gx = (float4)(0);
	float4 Gy = Gx;
	
	int c = x + y * width;

	if( x >= 1 && x < (width-1) && y >= 1 && y < (height - 1))
	{
		float4 i00 = convert_float4(inputImage[c - 1 - width]);
		float4 i10 = convert_float4(inputImage[c - width]);
		float4 i20 = convert_float4(inputImage[c + 1 - width]);
		float4 i01 = convert_float4(inputImage[c - 1]);
		float4 i11 = convert_float4(inputImage[c]);
		float4 i21 = convert_float4(inputImage[c + 1]);
		float4 i02 = convert_float4(inputImage[c - 1 + width]);
		float4 i12 = convert_float4(inputImage[c + width]);
		float4 i22 = convert_float4(inputImage[c + 1 + width]);

		// Sobel
		/*	 X				 Y
		{ 1  2  1}		{ 1  0 -1}
		{ 0  0  0}		{ 2  0 -2}
		{-1 -2 -1}		{ 1  0 -1}
		*/
		// Sk³adowe gradientu
		Gx = i00 + (float4)(2) * i10 + i20 - i02  - (float4)(2) * i12 - i22;
		Gy = i00 - i20  + (float4)(2) * i01 - (float4)(2) * i21 + i02  -  i22;
		int angle = gradToAngle(Gx.x, Gy.x);

		// Obraz wyjœciowy (konturów)
		// Modu³ gradientu c = sqrt(a^2 + b^2)
		//outputImage[c] = convert_uchar4(hypot(Gx, Gy)/(float4)(1));			
		outputImage[c] = inputImage[c];
		// Tablica k¹tów
		outputDir[c] = angle;
	}
}

__kernel void nms_dt(__global int* inputDir, __global uchar4* inputImage, __global uchar4* outputImage, __global int* tresh)
{
	uint x = get_global_id(0);
    uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);

	int c = x + y * width;
	float4 white = {255, 255, 255, 255};
	float4 blue = {1, 0, 0 , 0};
	float4 red = {0, 0, 0, 0};


	outputImage[c] = inputImage[c];

	/*if( x >= 1 && x < (width-1) && y >= 1 && y < (height - 1))
	{
		float4 i00 = convert_float4(inputImage[c - 1 - width]);
		float4 i10 = convert_float4(inputImage[c - width]);
		float4 i20 = convert_float4(inputImage[c + 1 - width]);
		float4 i01 = convert_float4(inputImage[c - 1]);

		float4 i11 = convert_float4(inputImage[c]);

		float4 i21 = convert_float4(inputImage[c + 1]);
		float4 i02 = convert_float4(inputImage[c - 1 + width]);
		float4 i12 = convert_float4(inputImage[c + width]);
		float4 i22 = convert_float4(inputImage[c + 1 + width]);
		

		if( inputDir[c] == 0 )
		{
			if(( i11.x > i10.x ) && ( i11.x > i12.x ))
			{
				if (i11.x > tresh[1]) //silna
					outputImage[c] = convert_uchar4(white);
				else if (i11.x < tresh[0])
					outputImage[c] = convert_uchar4(red);
				else	// slaba
					outputImage[c] = convert_uchar4(blue);
			}
			else		
				outputImage[c] = convert_uchar4(red);
		}
		if(inputDir[c] == 45)
		{
			if(( i11.x > i00.x ) && ( i11.x > i22.x ))
			{
				if (i11.x > tresh[1]) //silna
					outputImage[c] = convert_uchar4(white);
				else if (i11.x < tresh[0])
					outputImage[c] = convert_uchar4(red);
				else	// slaba
					outputImage[c] = convert_uchar4(blue);
			}
			else		
				outputImage[c] = convert_uchar4(red);			
		}
		if(inputDir[c] == 90)
		{ 
			if(( i11.x > i01.x ) && ( i11.x > i21.x ))
			{
				if (i11.x > tresh[1]) //silna
					outputImage[c] = convert_uchar4(white);
				else if (i11.x < tresh[0])
					outputImage[c] = convert_uchar4(red);
				else	// slaba
					outputImage[c] = convert_uchar4(blue);
			}
			else		
				outputImage[c] = convert_uchar4(red);
		}
		if(inputDir[c] == 135)
		{
			if(( i11.x > i02.x ) && ( i11.x > i20.x ))
			{			
				if (i11.x > tresh[1]) //silna
					outputImage[c] = convert_uchar4(white);
				else if (i11.x < tresh[0])
					outputImage[c] = convert_uchar4(red);
				else	// slaba
					outputImage[c] = convert_uchar4(blue);
			}
			else		
				outputImage[c] = convert_uchar4(red);
		}
	}*/
}

__kernel void blob(__global uchar4* inputImage, __global uchar4* outputImage)
{
	uint x = get_global_id(0);
    uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);

	int c = x + y * width;
	outputImage[c] = inputImage[c];
	/*
	if( x >= 1 && x < (width-1) && y >= 1 && y < (height - 1))
	{
		float4 i00 = convert_float4(inputImage[c - 1 - width]);
		float4 i10 = convert_float4(inputImage[c - width]);
		float4 i20 = convert_float4(inputImage[c + 1 - width]);
		float4 i01 = convert_float4(inputImage[c - 1]);

		float4 i11 = convert_float4(inputImage[c]);

		float4 i21 = convert_float4(inputImage[c + 1]);
		float4 i02 = convert_float4(inputImage[c - 1 + width]);
		float4 i12 = convert_float4(inputImage[c + width]);
		float4 i22 = convert_float4(inputImage[c + 1 + width]);


		if ((i11.x == 255) && (i11.z == 255))
		{
			outputImage[c] = convert_uchar4((float4)(255));
			// sprawdzam s¹siadów
			if (i00.x == 1){
				outputImage[c - 1 - width] = convert_uchar4((float4)(255));
				inputImage[c - 1 - width] = convert_uchar4((float4)(255));
			}
			if (i10.x == 1){
				outputImage[c - width] = convert_uchar4((float4)(255));
				inputImage[c - width] = convert_uchar4((float4)(255));
			}
			if (i20.x == 1){
				outputImage[c + 1 - width] = convert_uchar4((float4)(255));
				inputImage[c + 1 - width] = convert_uchar4((float4)(255));
			}
			if (i01.x == 1){
				outputImage[c - 1] = convert_uchar4((float4)(255));
				inputImage[c - 1] = convert_uchar4((float4)(255));
			}
			if (i21.x == 1){
				outputImage[c + 1] = convert_uchar4((float4)(255));
				inputImage[c + 1] = convert_uchar4((float4)(255));
			}
			if (i02.x == 1){
				outputImage[c - 1 + width] = convert_uchar4((float4)(255));
				inputImage[c - 1 + width] = convert_uchar4((float4)(255));
			}
			if (i12.x == 1){
				outputImage[c + width] = convert_uchar4((float4)(255));
				inputImage[c + width] = convert_uchar4((float4)(255));
			}
			if (i22.x == 1){
				outputImage[c + 1 + width] = convert_uchar4((float4)(255));
				inputImage[c + 1 + width] = convert_uchar4((float4)(255));
			}
		}
	}*/
}

