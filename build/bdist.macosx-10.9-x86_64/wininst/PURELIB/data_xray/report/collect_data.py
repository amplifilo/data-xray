from data_xray.file_io import FindScans
from data_xray.modules import *
from data_xray.report import SummaryPPT

#%%
topdir = os.getcwd()
print(topdir)

#%%
fdict = FindScans(topdir,'sxm')

#%%
pres = SummaryPPT(fdict=fdict)