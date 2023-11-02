import cProfile
import pstats
import testing
from cayley import*
import cProfile


# testing.ex4(n=4, k=4)
# Profile your function and save the result to 'profile_output.pstats'
cProfile.run('testing.runTimeIntegrationex4(n=10, k=10, cay=cay1)', 'profile_output.pstats')