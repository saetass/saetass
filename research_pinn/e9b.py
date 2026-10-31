import sys, time, os, numpy as np
sys.path.insert(0,'/home/cpozogonzalez/saetass/validation/cases'); sys.path.insert(0,'/home/cpozogonzalez/saetass/validation')
os.environ['MPLBACKEND']='Agg'
import adiabatic_shock_acceleration as c
from saetass.cli import set_log_level
set_log_level("ERROR")
for cells in (10,20):
    a=time.perf_counter(); r,p,psi,shock=c.run(cells=cells); t=time.perf_counter()-a
    print(f'SAETASS cells {cells}: {t:.1f}s slope {c.fitted_slope(p,psi[:,shock]):.4f}, theory {c.slope_spherical(4.0):.4f}',flush=True)
