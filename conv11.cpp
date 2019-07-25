#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<vector>
#include<stdio.h>
#include<time.h>
#include<string>
#include"file_manager.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


/*double Activationfunc(double input)
{	
	input=1/(1+exp(1-input));
	return input;
}*/


Mat conv_cal(int c,int c_cols,int c_rows,int d,int cha,int f_size,int s,Mat result,int ***pad,int **f,int ***rr,int sum)
{	
	clock_t begin=clock();	
	for(int k=0;k<cha;k++)
		{
		for(c=0;c<c_rows;c++)
		{
			for(d=0;d<c_cols;d++)
			{
			rr[k][c][d]=0;			
			}
		}
		}
	

	if(sum>1)
	{
	for(int k=0;k<cha;k++){
		for(int SC=0;SC<c_rows;SC++){		
			for(int SR=0;SR<c_cols;SR++){		
				for(int c=0;c<f_size;c++)
				{
					for(int d=0;d<f_size;d++)
					{	
				rr[k][SC][SR]+=(pad[k][c+SC*s][d+SR*s]*f[c][d]/sum);
					}
				}
				result.at<cv::Vec3b>(SC,SR)[k]=rr[k][SC][SR];
		}
	}
	}
	
	clock_t end=clock();
	double elapsed_secs=double(end-begin)/CLOCKS_PER_SEC;
	printf("Time_check %f\n",elapsed_secs);
	}
	else if(sum==0){
	for(int k=0;k<cha;k++){
		for(int SC=0;SC<c_rows;SC++){		
			for(int SR=0;SR<c_cols;SR++){		
				for(int c=0;c<f_size;c++)
				{
					for(int d=0;d<f_size;d++)
					{	
					rr[k][SC][SR]+=pad[k][c+SC*s][d+SR*s]*f[c][d];
					}
				}
				if(rr[k][SC][SR]<0)
				rr[k][SC][SR]=0;
				else if(rr[k][SC][SR]>255)
				rr[k][SC][SR]=255;
				result.at<cv::Vec3b>(SC,SR)[k]=rr[k][SC][SR];
			}
		}
	}
		
	clock_t end=clock();
	double elapsed_secs=double(end-begin)/CLOCKS_PER_SEC;
	printf("Time_check %f\n",elapsed_secs);
	}
		
	imshow("test1.jpg",result);
	//imwrite("blur_f5_s2_p10.jpg",result);
		waitKey(0);		
		return result;
}
void Max_pooling(int c,int M_cols,int M_rows,int d,int cha,int f_size,int s,Mat result2,Mat image,int **f)
{	
	clock_t begin=clock();		
	for(int k=0;k<cha;k++){
		for(int c=0;c<M_rows;c++){
			for(int d=0;d<M_cols;d++){
			result2.at<cv::Vec3b>(c,d)[k]=0;			
			}
		}
	}
	for(int k=0;k<cha;k++){
		for(int SC=0;SC<M_rows;SC++){		
			for(int SR=0;SR<M_cols;SR++)
			{
			int max=0;		
				for(int c=0;c<f_size;c++)
				{
					for(int d=0;d<f_size;d++)
					{	
					if(image.at<cv::Vec3b>(c+SC*s,d+SR*s)[k]>max)
					{
						max=image.at<cv::Vec3b>(c+SC*s,d+SR*s)[k];
					}
					}
				}
			result2.at<cv::Vec3b>(SC,SR)[k]=max;		
			}
		}
	}
clock_t end=clock();
	double elapsed_secs=double(end-begin)/CLOCKS_PER_SEC;
	printf("Time_check %f\n",elapsed_secs);
	
		imshow("test2.jpg",result2);
		imwrite("Max_s=1.jpg",result2);
		waitKey(0);
		return;
}


int main(){
	int i,image_size,c_cols,c_rows,a,b,c,d,e,choose,M_cols,M_rows;	
	//int ***w;
	

	int ***rr;
	int ***pad;
	int cha=3;
	int f_size;//{{1,2,1},{2,4,2},{1,2,1}};X
	int s=1;
	int p_size=1;
	Mat image=imread("ima.jpg",IMREAD_COLOR);  //image read
	
	if(image.empty())
	{
		cout<<"Could not open"<<endl;
		return -1;	
	}
	int x = image.cols;
	int y = image.rows; 
	printf("어떤 방법? 1.convolution 2.Max_pooling:");
	scanf("%d",&choose);
	printf("만들고 싶은 f배열의 원소의 수:");
	scanf("%d",&f_size);
	/*printf("만들고 싶은 channel:");
	scanf("%d",&cha);
	

	printf("stride:");
	scanf("%d",&s);
	printf("padding:");
	scanf("%d",&p_size);*/
	c_cols=((x-f_size+2*p_size)/s)+1;
	c_rows=((y-f_size+2*p_size)/s)+1;
	M_cols=((x-f_size)/s)+1;
	M_rows=((y-f_size)/s)+1;
	Mat result(c_rows,c_cols,image.type());
	Mat result2(M_rows,M_cols,image.type());
	//동적할당	
	//f = (int**)malloc(f_size*sizeof(int*));	
	rr = (int***)malloc(cha*sizeof(int**));	
	pad = (int***)malloc(cha*sizeof(int**));
	
	/*w = (int***)malloc(cha*sizeof(int**));							
	for(i=0;i<cha;i++)
	{						
	*(w+i) = (int**)malloc(image_size*sizeof(int*));						
	for(int j=0;j<image_size;j++)
	{						
	*(*(w+i)+j) = (int*)malloc(image_size*sizeof(int));
	}
	}*/

	//for(i=0;i<f_size;i++)
	//{						
	//*(f+i) = (int*)malloc(f_size*sizeof(int));
	//}

	for(i=0;i<cha;i++)
	{						
	*(rr+i) = (int**)malloc(c_rows*sizeof(int*));							
	for(int j=0;j<c_rows;j++)
	{						
	*(*(rr+i)+j) = (int*)malloc(c_cols*sizeof(int));
	}
	}

	for(i=0;i<cha;i++)
	{						
	*(pad+i)=(int**)malloc((y+2*p_size)*sizeof(int*));			
	for(int j=0;j<y+2*p_size;j++)
	{						
	*(*(pad+i)+j) = (int*)malloc((x+2*p_size)*sizeof(int));
	}
	}
	vector<vector<vector<double> > > filter = read_filter_all();

	int ** f=convert_vtoa(filter[1]); 
	//filter 
	int sum=0;
	printf("행렬 값2을 입력!:\n");
	for(a=0;a<f_size;a++)  //필터는 칠판에 있는거 만들어주기
	{
		for(b=0;b<f_size;b++)
		{
		//scanf("%d",&f[a][b]);
		sum=sum+f[a][b];
		}
	}
	/*if(sum>1){
	for(a=0;a<f_size;a++) 
	{
		for(b=0;b<f_size;b++)
		{
		f[a][b]=f[a][b]/sum;      //필터의 값이 너무커지면 ~질
					//blur=흐려지는거
					//1보다 2가 더흐림
					//edge detector 선만 따는거
					
		}
	}
}*/
	for(int k=0;k<cha;k++)
	{
		for(int i=0;i<y+2*p_size;i++)
		{
			for(int j=0;j<x+2*p_size;j++)
			{
			pad[k][i][j]=0;			
			}
		}
	}
	/*printf("행렬 값1을 입력!:");
	for(int k=0;k<cha;k++)
	{	
	for(a=0;a<image_size;a++)
	{
		for(b=0;b<image_size;b++)
		{
		scanf("%d",&w[k][a][b]);

		}
	}
	}*/

	for(int k=0;k<cha;k++){
		for(int i=0;i<y;i++){
			for(int j=0;j<x;j++){			
			pad[k][i+p_size][j+p_size]=image.at<cv::Vec3b>(i,j)[k];
			}
		}
	}
	/*for(int k=0;k<cha;k++){
		for(int i=0;i<y+2*p_size;i++){
			for(int j=0;j<x+2*p_size;j++){
			printf("%d",pad[k][i][j]);
			}printf("\n");
		}
	}
*/



	if(choose==1)
	conv_cal(c,c_cols,c_rows,d,cha,f_size,s,result,pad,f,rr,sum);
	else	
	Max_pooling(c,M_cols,M_rows,d,cha,f_size,s,result2,image,f);

	
	//double edge[] ={-1,-1,-1,-1,8,-1,-1,-1,-1};
	Mat efilter = (Mat_<float>(3,3) << -1,-1,-1,-1,8,-1,-1,-1,-1); 
	//Mat efilter = (Mat_<float>(3,3) << 0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625); 
	//Mat efilter = (Mat_<float>(3,3) << 1,2,1,2,4,2,1,2,1); 
	Mat test(c_rows,c_cols,image.type());
	filter2D(image,test,-1,efilter,Point(-1,-1),0,0);
	
	Mat result3(c_rows,c_cols,image.type());
	result3=test-result;
	//result3=result-test;
	namedWindow("result3",WINDOW_AUTOSIZE);
	imshow("result3",result3);
	//imshow("test",test);
	waitKey(0);

	for(int ch=0;ch<cha;ch++)
	{
		for(int i=0;i<c_rows;i++)
		{
			for(int j=0;j<c_cols;j++)
			{
			printf("%d",result3.at<Vec3b>(i,j)[ch]);
			//printf("%d",test.at<Vec3b>(i,j)[ch]);
			}
		}
	}
		
/*  for(i=0;i<cha;i++)
	{
	for(int j=0;j<image_size;j++)
	{
	free(*(*(w+i)+j));
	}				//p[i]=i;
	free(*(w+i));
	}free(w);
*/
	for(i=0;i<f_size;i++)
	free(*(f+i));				//p[i]=i;
	free(f);

	for(i=0;i<cha;i++)
	{
	for(int j=0;j<c_rows;j++)
	{
	free(*(*(rr+i)+j));
	}				//p[i]=i;
	free(*(rr+i));
	}free(rr);
	
	for(i=0;i<cha;i++)
	{
	for(int j=0;j<y+2*p_size;j++)
	{
	free(*(*(pad+i)+j));
	}				//p[i]=i;
	free(*(pad+i));
	}free(pad);
	return 0;
}
