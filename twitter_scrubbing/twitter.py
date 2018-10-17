import os

os.environ['R_HOME'] = r'C:\Program Files\R\R-3.5.1' 
os.environ['R_USER'] = r'C:\Users\Xue Yao\AppData\Local\Programs\Python\Python36-32\Lib\site-packages\rpy2' 

import rpy2.robjects as robjects

directory = r'C:\Users\Xue Yao\Documents\News Scrubbing\Twitter Scrubbing\twitter.R'
r_source = robjects.r['source']
r_source(directory)
print ('r script finished running')