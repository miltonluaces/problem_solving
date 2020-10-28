# Class TSCat. Transforms numerical time series in categorical.

class TSCat:
    
    # Fields 
    # ------------------------------------------------------------------------------------
    ranges = None
 
    # Constructor
    # ------------------------------------------------------------------------------------
    def __init__(self, ranges):
        self.ranges = ranges
        return

    # Methods 
    # ------------------------------------------------------------------------------------
    
    # Variable binning: x, numeric variable. returns category based in ranges
    def BinVar(self, x):
        for key,val in self.ranges.items():
            if x<=key : return val
        return -1

    # Time series binning: ts, time series. returns categorical time series based in ranges
    def BinTS(self, ts):
        binTs = []
        for i in range(0,len(ts)):
            binTs.append(self.BinVar(ts[i]))
        return binTs
        
        

