import threading
import logging
import TSAnalysis.TsIntegrals as ti

class MThOptimIrpe(threading.Thread):

        def __init__(self, name=None, ts=None, decRate=0.999, trace=False):
            threading.Thread.__init__(self, group=None, target=None, name=name)
            self.ts=ts
            self.decRate=decRate
            self.trace=trace
            self.output = None
        
        def run(self):
            self.output = ti.OptimIrpe(ts=self.ts, decRate=self.decRate, trace=self.trace)
            #logging.debug('running ', self.name)
            #print('running ', self.name)

