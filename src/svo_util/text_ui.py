'''
Created on Dec 10, 2012
@author: Stephen O'Hara

Utility functions for various text-mode user interface
functions. Typically things like showing a dotted progress
bar during the iterations of some batch process, etc.
'''
import sys

def print_progress(cur, total):
    '''
    This function can be called in a processing loop
    to print out a progress indicator represented
    by up to 10 lines of dots, where each line 
    represents completion of 10% of the total iterations.
    @param cur: The current value (integer) of the iteration/count.
    @param total: The total number of iterations that must occur.
    '''
    one_line = 40 if total < 400 else round( total / 10.0 )
    one_dot = 1 if one_line / 40.0 < 1 else round( one_line / 40.0)    
    if (cur+1)%one_line == 0:
        print ' [%d]'%(cur+1)
    elif (cur+1)%one_dot == 0:
        print '.',
        sys.stdout.flush()    
    if cur+1 == total: print ""