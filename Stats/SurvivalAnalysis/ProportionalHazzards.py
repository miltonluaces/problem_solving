import rpy2
import lifelines
import plotly
import pandas
from rpy2.robjects.packages import importr

# Load python libraries
import numpy as np
import pandas as pd
import lifelines as ll

from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from IPython.display import HTML
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls   
from plotly.graph_objs import *

from pylab import rcParams
rcParams['figure.figsize']=10, 5


Rggplot2 = importr('ggplot2')
Rsurvival = importr('survival')
devtools = importr('devtools')
Rplotly = importr('plotly')
RIRdisplay = importr('IRdisplay')
RKmSurv = importr('KMsurv')

# Authenticate to plotly's api using your account
#py = plotly("rmdk", "0sn825k4r8")

#tg = RKmSurv('tongue')
pi = ro.r('pi')
tg = ro.r('tongue')
#tg = ro.r('''data(tongue)''')
print(tg)
#pull tongue
#tongue = pandas2ri.ri2py_dataframe(tg)


