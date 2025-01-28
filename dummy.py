import pandas as pd

data  = pd.read_csv(r'C:\Users\k67885\Documents\EyeGaze\output\GameVideo_Horizontal_Medium1738074854.avi.csv')

#print(data)
#print(data.columns)
data.columns = data.columns.str.replace(' ','_')

coulmns = ['Frame','Time_(s)','Screen_X_(px)', 'Screen_Y_(px)', '']
df =  data[coulmns]
print(df.describe())