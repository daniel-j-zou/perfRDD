#This file is to check if the stata processing worked

import os
import pandas as pd
import subprocess

os.chdir('/Users/lukat/PycharmProjects/GPA_STATA_Data')

filepath_dta = os.path.join(os.getcwd(), 'Dep_Data/AEJApp2008-0202_data', 'data_for_analysis.dta')
dta = pd.read_stata(filepath_dta, convert_categoricals=False)
processed_dta = pd.read_csv('Dep_Data/final_processed_data.csv')

print(dta.columns)
print(processed_dta.columns)



# if dta.equals(processed_dta):
#     print("They Match")
#
# else:
#     print("They Doesn't Match")

