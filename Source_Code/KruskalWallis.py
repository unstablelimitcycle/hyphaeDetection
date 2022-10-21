#Kruskal Wallis H Test for Yeast
from scipy import stats
import pandas

#Across all strains
#Read and store data from CSV
df1 = pandas.read_csv('All Strains 0uM Data.csv', index_col = 'Strain_picNum') #Changed index to Image #, name of first column, but do we even need that?
#Pretty much only need the h-measures at different QS concentrations

#Convert Data Frame to Array
DataGroup1 = df1.to_numpy()

#Read and store data from CSV
df2 = pandas.read_csv('All Strains 100uM Data.csv', index_col = 'Strain_picNum') #Changed index to Image #, name of first column, but do we even need that?
#Pretty much only need the h-measures at different QS concentrations

#Convert Data Frame to Array
DataGroup2 = df2.to_numpy()

#Read and store data from CSV
df3 = pandas.read_csv('All Strains 200uM Data.csv', index_col = 'Strain_picNum') #Changed index to Image #, name of first column, but do we even need that?
#Pretty much only need the h-measures at different QS concentrations

#Convert Data Frame to Array
DataGroup3 = df3.to_numpy()

group1 = DataGroup1[:,0]
group2 = DataGroup2[:,0]
group3 = DataGroup3[:,0]

result = stats.kruskal(group1, group2, group3)
print(result)
#Within Strains across concentrations
