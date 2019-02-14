"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Implements ProSRS algorithm.
"""
#TODO: change 'processes=' to 'nodes=' for pool initialization.
#TODO: change 'wgt_expon' to 'gamma'. "wgt_expon = - gamma". Change rbf_wgt() and optimizer function accordingly.