#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xarray as xr
import numpy as np
#    import matplotlib.pyplot as plt
import pandas as pd
import glob
import warnings
warnings.simplefilter("ignore")
import os

# In[ ]:


# This is the first version of a function to output data from the local cmip6
# archive as an xarray data set for analysis
# Currently designed to be given a model, experiment, and variable (plus table)
# as inputs.  Will find the number of ensemble members available.
# If unspecified will use all ensemble members
# Future updates will add functionality
def local_cmip6_read(exp,model,variable,freq,no_esm=False):
    
    general_folder='/home/disk/eos9/cmip6/'
    ff=general_folder + exp +'/' + model + '/' + variable +'/'
    fpre=ff+variable + '_' + freq + '_' + model + '_'+ exp + '_r'
    if not no_esm:
        no_ens=list_ensembles(exp,model,variable,freq)
    else:
        no_ens=no_esm
    #use previously generated list of ensemble members to create a new dataset over ensemble members
    ens_list=[]
    for e in no_ens:
        rz=str(e)
        ens= 'ens%d' % e
        mfl=fpre + rz +'i1p1*_gn*'
        
        ens_list.append(xr.open_mfdataset(mfl))
    ds=xr.concat(ens_list, dim='ensemble')
#    ds=mfl
    return ds 

def local_cmip6_read_ag(exp,model,variable,freq,no_esm=False):
    
    general_folder='/home/disk/eos9/cmip6/'
    ff=general_folder + exp +'/' + model + '/' + variable +'/'
    fpre=ff+variable + '_' + freq + '_' + model + '_'+ exp + '_r'
    if not no_esm:
        no_ens=list_ensembles(exp,model,variable,freq)
    else:
        no_ens=no_esm
    #use previously generated list of ensemble members to create a new dataset over ensemble members
    ens_list=[]
    for e in no_ens:
        rz=str(e)
        ens= 'ens%d' % e
        mfl=fpre + rz +'i1p1f*'
        
        ens_list.append(xr.open_mfdataset(mfl))
    ds=xr.concat(ens_list, dim='ensemble')
#    ds=mfl
    return ds 

# In[7]:


#This function finds the number of ensemble members present in our archives
def find_num_ensembles(exp,model,variable,freq):
    general_folder='/home/disk/eos9/cmip6/'
    ff=general_folder + exp +'/' + model + '/' + variable +'/'
    #expanding to files in an experiment, model, variable
    files= [fi for fi in glob.glob(ff + variable + "_" + freq +"*")]
    #Now need to find the number of ensemble members.
    #All files have the same prefix and location pattern
    #so add up characters before r*i1p1f1 (can be modified later to account for other options)

    fpre=ff+variable + '_' + freq + '_' + model + '_'+ exp + '_r'
    #cut length of file name to just be r*i*p*f*
    fprel=len(fpre)
    filestrunc=[fi[fprel-1:fprel+8] for fi in files]
    filestrunc1=np.array(filestrunc)
    ens_name=np.unique(filestrunc1)
    no_ens=len(ens_name)
    return no_ens


#This function finds the number of ensemble members present in our archives
def find_models(exp,variable):
    general_folder='/home/disk/eos9/cmip6/'
    fe=general_folder + exp +'/'
    #find models (each model should have its own folder)
    amodels= [fi for fi in glob.glob(fe +"*")]
    #check if variable available in model
    models=[]
    for fi in amodels:
        fm= fi + '/'+ variable + '/'
        if os.path.isdir(fm):
            fprel=len(fe)
            mi=fi[fprel:]
            models.append(mi)
    return models

#This function finds the number of ensemble members present in our archives
def list_ensembles(exp,model,variable,freq):
    general_folder='/home/disk/eos9/cmip6/'
    ff=general_folder + exp +'/' + model + '/' + variable +'/'
    #expanding to files in an experiment, model, variable
    files= [fi for fi in glob.glob(ff + variable + "_" + freq + "*")]
    #Now need to find the number of ensemble members.
    #All files have the same prefix and location pattern
    #so add up characters before r*i1p1f1 (can be modified later to account for other options)

    fpre=ff+variable + '_' + freq + '_' + model + '_'+ exp + '_r'
    #cut length of file name to just be r*i*p*f*
    fprel=len(fpre)
    filestrunc=[fi[fprel-1:fprel+8] for fi in files]
    filestrunc1=np.array(filestrunc)
    ens_name=np.unique(filestrunc1)
    rnames=[]
    for e in ens_name:
        ens_name1=str(e)
        rnames1=ens_name1.find('i')
        a=e[1:rnames1]
        rnames.append(a)
    rnames1=list(map(int,rnames))
    return rnames1


