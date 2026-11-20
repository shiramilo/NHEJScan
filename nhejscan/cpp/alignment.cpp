//alignment.cpp

#include"alignment.h"
#include<vector>
#include<iostream>
#include<cmath>
#include<cstdio>

using namespace std;

alignment::alignment(vector<int> s1, vector<int> s2, int lin)
{  
  seq1=s1;
  seq2=s2;
  l=lin;
    
  L1=seq1.size()+1;
  L2=seq2.size()+1;
  
  S1=new int [L2];
  S2=new int [L2];
}

alignment::~alignment()
{
  delete[] S1;
  delete[] S2;
}

int alignment::score(int a, int b)
{
  if(a==b && a!=4) return 1;
  else return -9; 
}

int alignment::align()
{
  int *temp;
  int l_max=0;
  int S_max=0;
  int inf=1000;
  
  //i=0
  for(int j=0; j<L2; j++) 
  {
    if(j<l) { S1[j]=-j*9; }
    else { S1[j]=-inf; }
    //printf("%5d ",S1[j]);
  }
  
  //cout << endl;
  
  //i=1...L1
  for(int i=1; i<L1; i++)
  {
    //j=0
    if(i<l) { S2[0]=-i*9; }
    else { S2[0]=-inf; }
    //printf("%5d ",S2[0]);

    //j=1...L
    for(int j=1; j<L2; j++)
    {
      if(abs(i-j)<l)
      {
        S2[j]=S1[j-1]+score(seq1[i-1],seq2[j-1]);
        if(S2[j-1]-9>S2[j]) S2[j]=S2[j-1]-9;
        if(S1[j]-9>S2[j]) S2[j]=S1[j]-9;
        if(S2[j]>=S_max)
        {
          S_max=S2[j];
          if(max(i,j)>l_max) { l_max=max(i,j); }
        }
      }
      else if(abs(i-j)==l)
      {
        S2[j]=-inf;
      }
      //printf("%5d ",S2[j]);
    }  
  
    //cout << endl;
       
    //switch S1<->S2
    if(i<L1-1)
    {
      temp=S1;
      S1=S2;
      S2=temp;
    }
  }
  return l_max;  
}


