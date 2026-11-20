//alignment.cpp
#include<iostream>
//#include<sstream>
//#include<fstream>
#include"alignment.h"
#include<string>

vector<int> stov(string s)
{
  vector<int> v;
  for(int i=0; i<s.length(); i++)
  {
    v.push_back(atoi(s.substr(i,1).c_str()));
  }
  return v;
}

using namespace std;

int main(int argc, char* argv[])
{        
    string seq1_s=argv[1];
    string seq2_s=argv[2];
    int l=atoi(argv[3]);
    
    //cout << seq2_s << endl;
    //cout << seq1_s << endl;
    
    vector<int> seq1=stov(seq1_s);
    vector<int> seq2=stov(seq2_s);
    
    //align lrc local
    
    alignment al(seq1,seq2,l);
    cout << al.align();
}


