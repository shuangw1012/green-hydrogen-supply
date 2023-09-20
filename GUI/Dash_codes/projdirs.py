# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:32:04 2022

@author: Ahmad Mojiri
"""

import os,platform

 
connector = ['\\','/'][platform.system()=='Linux']

if platform.system()=='Linux':
    basedir = r'/home/ahmadmojiri/GreenH2/'
else:
    # basedir = os.path.realpath('..') + connector
    basedir = r'C:\\Nextcloud\\HILT-CRC---Green-Hydrogen\\'
    
    

datadir = basedir + "DATA%s" %connector
# modeldir = basedir + "modelling%spython%spackage%s" %(connector, connector, connector)
optdir = basedir + "MINIZINC%s" %(connector)
# figdir = basedir + "modelling%sfigures%s" %(connector, connector)
# paperdir = basedir + "Publications%spaper_1%s" %(connector, connector)
# resultsdir = datadir + "arbitrage%s" %connector