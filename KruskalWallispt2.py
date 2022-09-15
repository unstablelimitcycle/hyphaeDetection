#Kruskal Wallis H Test for Yeast
from scipy import stats
import pandas
import scikit_posthocs as sp

#Within Strains across concentrations
#Read and store data from CSV
df = pandas.read_csv('F2 Data.csv', index_col = 'Image #') 
x1 = df["Growth Index, 0uM"].mean()
x2 = df["Growth Index, 100uM"].mean()
x3 = df["Growth Index, 200uM"].mean()
df["Growth Index, 0uM"].fillna(x1, inplace = True)
df["Growth Index, 100uM"].fillna(x2, inplace = True)
df["Growth Index, 200uM"].fillna(x3, inplace = True)

#Convert Data Frame to Array
DataGroup = df.to_numpy()

group1 = DataGroup[:,0]
group2 = DataGroup[:,1]
group3 = DataGroup[:,2]
#print(group1)
data = [group1, group2,  group3]
KruskalWallisResult = stats.kruskal(group1, group2, group3)
DunnTestResult = sp.posthoc_dunn(data, p_adjust = 'bonferroni')
print(KruskalWallisResult)
print(DunnTestResult)