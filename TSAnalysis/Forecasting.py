# Imports

import pandas as pd
import numpy as np
import itertools as it
import warnings as wr
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa as smt
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from scipy.stats import norm
from datetime import datetime
import requests
from io import BytesIO


