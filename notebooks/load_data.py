def load_CMIP6data(experiments,source='',table='',variable='',grid_label=''):
#    %matplotlib inline

    import xarray as xr
    import intake
    import util 
    import sys
    
    if util.is_ncar_host():
        col = intake.open_esm_datastore("../catalogs/glade-cmip6.json")
    else:
        col = intake.open_esm_datastore("../catalogs/pangeo-cmip6.json")
    
    tables_all = col.df.table_id.unique()
    vars_all = col.df.variable_id.unique()
    grids_all = col.df.grid_label.unique()
    source_all = col.unique('source_id')

    ## Find the entries
    if table=='':
        table = 'Amon'
        print('Table will be set to "Amon" by default')
    elif table not in tables_all:
        sys.exit(['Table names can only be one of the following: ', tables_all])
        
    if variable=='':
        sys.exit('Please specify the name of the VARIABLE you want to extract. ')
    elif variable not in vars_all:
        sys.exit(['Variable names can only be one of the following: ', vars_all])
        
    if grid_label =='':
        sys.exit('Please specify the name of the GRID you want to extract. ')
    elif grid_label not in grids_all:
        sys.exit(['Grid_table can only be one of the following: ', grids_all])
        
    if source=='':
        source = source_all
        
    ## locate the models
    import pprint 
    uni_dict = col.unique(['source_id', 'experiment_id', 'table_id'])
    #pprint.pprint(uni_dict, compact=True)
    models = set(uni_dict['source_id']['values']) # all the models

    for experiment_id in [experiments]:
        query = dict(experiment_id=experiment_id, table_id=table, 
                     variable_id=variable, grid_label=grid_label)  
        cat = col.search(**query)
        models = models.intersection({model for model in cat.df.source_id.unique().tolist()})

    # ensure the CESM2 models are not included (oxygen was erroneously submitted to the archive)
    models = models - {'CESM2-WACCM', 'CESM2'}

    models = list(models)
    
    cat = col.search(experiment_id=experiments, table_id=table, 
                 variable_id=variable, grid_label=grid_label, source_id=models)
    
    ## load data
    dset_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True, 'decode_times': False}, 
                                cdf_kwargs={'chunks': {}, 'decode_times': False})
        
    return dset_dict
    
