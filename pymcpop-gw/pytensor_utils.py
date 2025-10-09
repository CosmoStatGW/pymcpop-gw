import time
import sys


def make_tqdm_callback(pbar):
    t0 = time.perf_counter()

    def cb(*args, **kwargs):
        """
        Supports:
          - PyMC >=5: (draw, tuning, chain)
          - Older patterns: (trace, draw) or (draw,)
        """
        draw = tuning = chain = None

        if len(args) >= 3:
            draw, tuning, chain = args[:3]
        elif len(args) == 2:
            # could be (trace, draw)
            draw = args[1]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        elif len(args) == 1:
            draw = args[0]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        else:
            draw = kwargs.get("draw", 0)
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)

        # update bar (throttle if you like)
        pbar.update(1)
        if (pbar.n % 25) == 0:
            phase = "warmup" if tuning else "sampling"
            rate = pbar.n / max(time.perf_counter() - t0, 1e-9)
            pbar.set_postfix_str(f"{phase} | {rate:5.1f} it/s", refresh=False)

    return cb





# Writes output both on std output and on log file
class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False
