//alignment.h
#ifndef alignmen_h
#define alignment_h

#include<vector>

using namespace std;

class alignment
{
  public:
    alignment(vector<int> s1, vector<int> s2, int l1);
    ~alignment();
    int score(int a, int b);
    int align();
        
  private:
    vector<int> seq1;
    vector<int> seq2;
    int l;
    int *S1;
    int *S2;
    unsigned long L1;
    unsigned long L2;
};

#endif

