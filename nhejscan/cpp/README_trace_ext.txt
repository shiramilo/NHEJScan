Calculating indel trace extensions: 

The algorithm calculates how far from their beginning two sequences can be aligned. For example, consider the two sequences:

1: ACATGAGGTCTGACTT
2: ACATGAGGACGTGTAC
   ++++++++--------

Here one would conclude that they can be aligned for the first 8 bases. The alignment is performed by the program align.cpp. The used class "alignment" is specified in alignment.cpp and alignment.h. As its first two input parameters, align.cpp takes two strings of the sequences to be aligned. They need to be mapped on integers according to the mapping scheme:

A -> 0
C -> 1
G -> 2
T -> 3
N -> 4

For example, the sequence ATGTCNT would map on the string 0323143. Align.cpp does incorporate mismatches and gaps in the alignment as specified in the manuscript. As third input parameter, align.cpp requires the length of the indel as this poses an upper limit on the number of allowed gaps in the alignment. The output of align.cpp is an integer specifying the length that the two sequences can be aligned for (in the above example this would be 8).

The trace extension of an indel is then calculated according to the procedure described in FIG. 1 of the MBE paper. This means first the sequences from species 1 and 2 would be aligned from point 3 towards direction 4. The command would be:

align TGATGT... TGACATGTGATGT... 7 

Here the sequences would of course have to be mapped onto integers 0,1,2,3,4 according to the above scheme (which i did not do so that it can be better compared with FIG.1). The output would be 3. You also have to decide how far you want the alignment to check, i.e. how long the two sequences are (indicated by the dots). It should be fine to restrict this to two or three times the indel length, unless you expect really long trace extensions.

Then you have to perform the same procedure for the alignment from point 3' towards point 4' in FIG.1. Notice that here the sequences have to be reversed. The command would be:

align GTACAGTCTC... GTACAGTGTACAGTCTC... 7

And the output will be 7. The trace extension of the indel is then the sum of both outputs, 7+3=10. 
