#!/usr/bin/env python
# coding: utf-8


import os, json, pickle

import numpy as np

from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import gp_signals
from enterprise.signals import gp_priors
from enterprise.signals import parameter
from enterprise.signals import selections

from enterprise_extensions import blocks

import cloudpickle

# # using `enterprise_extensions`
#
# `enterprise_extensions` provides "recipes" for commonly used functionality in `enterprise`.
#
# Lets build a 3 pulsar PTA that we could use to search for a gravitational wave background.

# ## load data

# In[2]:


# datafiles = {
#     "J1600-3053":{"par":"J1600-3053_EPTA_6psr.par", "tim":"J1600-3053_EPTA_6psr.tim"},
#     "J2241-5236":{"par":"J2241-5236_PPTA_dr2.par", "tim":"J2241-5236_PPTA_dr2.tim"},
#     "J2317+1439":{"par":"J2317+1439_NANOGrav_12y.par", "tim":"J2317+1439_NANOGrav_12y.tim"},
# }
#
# datadir = os.path.abspath("data")
#
#
# # In[3]:
#
#
# # load in each pulsar and append it to a list
# psrs = []
# for pname, fdict in datafiles.items():
#     pfile = os.path.join(datadir, fdict["par"])
#     tfile = os.path.join(datadir, fdict["tim"])
#     psrs.append(Pulsar(pfile, tfile))

datadir = os.path.abspath("../Day_1/data")

with open(datadir+'/viper_3psr.pkl','rb') as fin:
    psrs = pickle.load(fin)


# (once again, we can safely ignore these `tempo2` warnings)

# ## determine the PTA `Tspan`
# When building a `PTA` using data from multiple pulsars it helps to have a common Fourier basis for all of the pulsars' red noise (and common red noise, like GWB).  The easy way to do this is to use the total time-span of all data to set the Fourier frequencies.
#
# `enterprise.signals.gp_signals.FourierBasisGP` can use an intput `Tspan` to figure out the frequencies, and several functions in `enterprise_extensions` can too.

# In[4]:


# calculate the total Tspan
Tspan = np.max([pp.toas.max() - pp.toas.min() for pp in psrs])


# ## generate an enterprise `PTA` for all three pulsars for a CRN analysis
#
# Each pulsar needs a different noise model.  For CRN analysis it is common to fix the WN parameters based on previous single pulsar noise runs.
#
# To speed up the likelihood calculation we can use the `enterprise.signals.gp_signals.MarginalizingTimingModel`, which breaks the GP coefficient marginalization into two steps.  The linear timing model is analytically marginalized first.  This reduces the size of the matrices that must be inverted at each likelihood evaluation.  Only the Fourier Basis GPs (RN, DM, GWB, ...) contribute.
#
# We're going to use a spatially correlated common red noies model with a powerlaw spectrum as our GWB.
#
# Let's start by building the parts of the model that all pulsars will include:
#
# * timing model
# * red noise -- 30 frequency powerlaw -- `enterprise_extensions.blocks.red_noise_block`
# * GWB -- 15 frequency powerlaw, Hellings-Downs correlated -- `enterprise_extensions.blocks.common_red_noise_block`
#  * $\log_{10} A \rightarrow$ Uniform(-18, -13)
#  * $\gamma=13/3$

# In[6]:


# make the timing model signal
tm = gp_signals.MarginalizingTimingModel(use_svd=True)


# In[7]:


# make the RN signal
rn = blocks.red_noise_block(
    psd="powerlaw", components=30,
    Tspan=Tspan
)


# In[8]:


# make the GWB signal
gw = blocks.common_red_noise_block(
    psd="powerlaw", components=15,
    gamma_val=13/3,
    orf="hd",
    Tspan=Tspan
)

# make the Common Red Process signal
crn = blocks.common_red_noise_block(
    psd="powerlaw", components=15,
    gamma_val=13/3,
    orf=None,
    Tspan=Tspan
)


# Since each pulsar has a unique model, we'll store the three `SignalCollections` as a list.

# In[9]:


# empty list to store each pulsar's "signal" model
crn_sigs = []
gw_sigs = []



# ### generate an enterprise signal model for EPTA's J1600 pulsar
#
# In addition to the timing model, RN, and GWB, we need to include:
#
# * white noise -- fixed EFAC & EQUAD per backend (no ECORR)
# * DM variations -- 100 frequency powerlaw DM GP
#
# These are easy to do using `enterprise_extensions.blocks`.
#
# For a GWB analysis it is common to hold the white noise parameters (EFAC/EQUAD/ECORR) fixed to some known value (as determined by a single pulsar analysis.
# This reduces the number of parameters in the full PTA model.
# `enterprise` accomplishes this by using the `parameter.Constant` class.
# `enterprise_extensions.blocks.white_noise_block` has a boolean option to control this behavior.
# We'll use `vary=False` for **fixed** WN.

# In[10]:


# make the WN signal
wn = blocks.white_noise_block(vary=False, inc_ecorr=False, select="backend")


# In[11]:


# make the DM GP signal
dm = blocks.dm_noise_block(gp_kernel="diag", psd="powerlaw", components=100, Tspan=Tspan)


# In[12]:


# append J1600's SignalCOllection to the list
gw_sigs.append(tm + wn + rn + dm + gw)
crn_sigs.append(tm + wn + rn + dm + crn)


# ### generate an enterprise signal model for PPTA's J2241 pulsar
#
# In addition to the timing model, RN, and GWB, we need to include:
#
# * white noise -- fixed EFAC & EQUAD per backend (no ECORR)
# * DM variations -- 100 frequency powerlaw DM GP
# * band noise -- 30 frequency powerlaw in the 20cm band
#
# We can reuse the same `wn` and `dm` signals from before.
#
# To implement band noise we need a `enterprise.signal.selections.Selection`.
# A selection function takes the `dict` of TOA flags and flagvals as input.
# It returns a `dict` whose keys are the flagvals to select and mask (array of True/False) telling which TOAs have that flag.
#
# There's a built in `by_band` selection function, but that applies band noise to **all** bands.
# We only want to apply this model to TOAs in the 20cm band, so we need a selection function that returns a `dict` with one key and a mask for that flagval.

# In[13]:


def band_20cm(flags):
    """function to select TOAs in 20cm band (-B 20CM)"""
    flagval = "20CM"
    return {flagval: flags["B"] == flagval}

by_band_20cm = selections.Selection(band_20cm)


# There's no band noise block in `enterprise_extensions` but we can make a Fourier basis GP with the appropriate selection the old fashioned way!

# In[14]:


# band noise parameters
BN_logA = parameter.Uniform(-20, -11)
BN_gamma = parameter.Uniform(0, 7)

# band noise powerlaw prior
powlaw = gp_priors.powerlaw(log10_A=BN_logA, gamma=BN_gamma)

# make band noise signal (don't forget the name!)
bn = gp_signals.FourierBasisGP(
    powlaw,
    components=30,
    Tspan=Tspan,
    selection=by_band_20cm,
    name="band_noise"
)


# In[15]:


# append J2241's SignalCOllection to the list
gw_sigs.append(tm + wn + rn + bn + dm + gw)
crn_sigs.append(tm + wn + rn + bn + dm + crn)


# ### generate an enterprise signal model for NANOGrav's J2317 pulsar
#
# In addition to the timing model, RN, and GWB, we need to include:
#
# * white noise -- fixed EFAC, EQUAD, **and ECORR** per backend
#
# Remember there is no DM variations model, because DMX is already in the timing model for NANOGrav's 12.5yr data release

# In[16]:


# make WN signal (now with ECORR!)
wn_ec = blocks.white_noise_block(vary=False, inc_ecorr=True, select="backend")


# In[17]:


# append J2317's SignalCOllection to the list
gw_sigs.append(tm + wn_ec + rn + gw)
crn_sigs.append(tm + wn_ec + rn + crn)


# ## put the three pulsars together into a `PTA` object
#
# We can instantiate a PTA object with a list of three pulsar models.
# We simply feed each `Pulsar` to its `SignalCollection`, and then pass the whole list of instantiated models to `signal_base.PTA`.

# In[18]:


pta_gw = signal_base.PTA([ss(pp) for ss,pp in zip(gw_sigs, psrs)])
pta_crn = signal_base.PTA([ss(pp) for ss,pp in zip(crn_sigs, psrs)])

# ### load noise dictionary
#
# At this point we never actually told `enterprise` what to use for the fixed the WN parameters.
# We can use `PTA.set_default_params` to pass in the correct WN values from a `dict`.
#
# First we'll load the dictionary, which is stored as a `.json` file in the `data/` directory

# In[19]:


nfile = os.path.join(datadir, "viper_3psr_noise.json")
with open(nfile, "r") as f:
    noisedict = json.load(f)

# set the fixed WN params
pta_gw.set_default_params(noisedict)
pta_crn.set_default_params(noisedict)

with open('./vandy_3psr_fullpta_gwb.pkl','wb') as fout:
    cloudpickle.dump(pta_gw,fout)

with open('./vandy_3psr_fullpta_crn.pkl','wb') as fout:
    cloudpickle.dump(pta_crn,fout)
