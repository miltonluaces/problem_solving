import unittest
import numpy as np
import Misc.Tables as vmt


class TestMisc(unittest.TestCase):

    
    def test10_Tables(self):
        columns = ('Last', 'High', 'Low', 'Chg', 'Chg%', 'Time', 'Fcst')
        rows = ['Gold', 'Silver', 'Copper', 'Alumin']
        data = np.random.randint(10,90, size=(len(rows), len(columns)))
        vmt.ShowTable(data, columns, rows)
        vmt.ShowTable(data, columns)

