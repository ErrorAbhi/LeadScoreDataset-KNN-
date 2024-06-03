### Importing Necessary Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Reading the Dataset


```python
df = pd.read_csv(r'C:\Users\Administrator\Downloads\Lead Scoring Assignment\Leads.csv')
```

### Inspection of Dataset


```python
# to know the dataset
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prospect ID</th>
      <th>Lead Number</th>
      <th>Lead Origin</th>
      <th>Lead Source</th>
      <th>Do Not Email</th>
      <th>Do Not Call</th>
      <th>Converted</th>
      <th>TotalVisits</th>
      <th>Total Time Spent on Website</th>
      <th>Page Views Per Visit</th>
      <th>...</th>
      <th>Get updates on DM Content</th>
      <th>Lead Profile</th>
      <th>City</th>
      <th>Asymmetrique Activity Index</th>
      <th>Asymmetrique Profile Index</th>
      <th>Asymmetrique Activity Score</th>
      <th>Asymmetrique Profile Score</th>
      <th>I agree to pay the amount through cheque</th>
      <th>A free copy of Mastering The Interview</th>
      <th>Last Notable Activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7927b2df-8bba-4d29-b9a2-b6e0beafe620</td>
      <td>660737</td>
      <td>API</td>
      <td>Olark Chat</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>No</td>
      <td>Select</td>
      <td>Select</td>
      <td>02.Medium</td>
      <td>02.Medium</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>No</td>
      <td>No</td>
      <td>Modified</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2a272436-5132-4136-86fa-dcc88c88f482</td>
      <td>660728</td>
      <td>API</td>
      <td>Organic Search</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>5.0</td>
      <td>674</td>
      <td>2.5</td>
      <td>...</td>
      <td>No</td>
      <td>Select</td>
      <td>Select</td>
      <td>02.Medium</td>
      <td>02.Medium</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>No</td>
      <td>No</td>
      <td>Email Opened</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8cc8c611-a219-4f35-ad23-fdfd2656bd8a</td>
      <td>660727</td>
      <td>Landing Page Submission</td>
      <td>Direct Traffic</td>
      <td>No</td>
      <td>No</td>
      <td>1</td>
      <td>2.0</td>
      <td>1532</td>
      <td>2.0</td>
      <td>...</td>
      <td>No</td>
      <td>Potential Lead</td>
      <td>Mumbai</td>
      <td>02.Medium</td>
      <td>01.High</td>
      <td>14.0</td>
      <td>20.0</td>
      <td>No</td>
      <td>Yes</td>
      <td>Email Opened</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 37 columns</p>
</div>




```python
# to know the shape of dataset / # of rows and # of columns
df.shape
```




    (9240, 37)




```python
# to know the datatypes and null in every columns
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9240 entries, 0 to 9239
    Data columns (total 37 columns):
     #   Column                                         Non-Null Count  Dtype  
    ---  ------                                         --------------  -----  
     0   Prospect ID                                    9240 non-null   object 
     1   Lead Number                                    9240 non-null   int64  
     2   Lead Origin                                    9240 non-null   object 
     3   Lead Source                                    9204 non-null   object 
     4   Do Not Email                                   9240 non-null   object 
     5   Do Not Call                                    9240 non-null   object 
     6   Converted                                      9240 non-null   int64  
     7   TotalVisits                                    9103 non-null   float64
     8   Total Time Spent on Website                    9240 non-null   int64  
     9   Page Views Per Visit                           9103 non-null   float64
     10  Last Activity                                  9137 non-null   object 
     11  Country                                        6779 non-null   object 
     12  Specialization                                 7802 non-null   object 
     13  How did you hear about X Education             7033 non-null   object 
     14  What is your current occupation                6550 non-null   object 
     15  What matters most to you in choosing a course  6531 non-null   object 
     16  Search                                         9240 non-null   object 
     17  Magazine                                       9240 non-null   object 
     18  Newspaper Article                              9240 non-null   object 
     19  X Education Forums                             9240 non-null   object 
     20  Newspaper                                      9240 non-null   object 
     21  Digital Advertisement                          9240 non-null   object 
     22  Through Recommendations                        9240 non-null   object 
     23  Receive More Updates About Our Courses         9240 non-null   object 
     24  Tags                                           5887 non-null   object 
     25  Lead Quality                                   4473 non-null   object 
     26  Update me on Supply Chain Content              9240 non-null   object 
     27  Get updates on DM Content                      9240 non-null   object 
     28  Lead Profile                                   6531 non-null   object 
     29  City                                           7820 non-null   object 
     30  Asymmetrique Activity Index                    5022 non-null   object 
     31  Asymmetrique Profile Index                     5022 non-null   object 
     32  Asymmetrique Activity Score                    5022 non-null   float64
     33  Asymmetrique Profile Score                     5022 non-null   float64
     34  I agree to pay the amount through cheque       9240 non-null   object 
     35  A free copy of Mastering The Interview         9240 non-null   object 
     36  Last Notable Activity                          9240 non-null   object 
    dtypes: float64(4), int64(3), object(30)
    memory usage: 2.6+ MB
    


```python
# for descriptive analysis of the dataset
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lead Number</th>
      <th>Converted</th>
      <th>TotalVisits</th>
      <th>Total Time Spent on Website</th>
      <th>Page Views Per Visit</th>
      <th>Asymmetrique Activity Score</th>
      <th>Asymmetrique Profile Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9240.000000</td>
      <td>9240.000000</td>
      <td>9103.000000</td>
      <td>9240.000000</td>
      <td>9103.000000</td>
      <td>5022.000000</td>
      <td>5022.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>617188.435606</td>
      <td>0.385390</td>
      <td>3.445238</td>
      <td>487.698268</td>
      <td>2.362820</td>
      <td>14.306252</td>
      <td>16.344883</td>
    </tr>
    <tr>
      <th>std</th>
      <td>23405.995698</td>
      <td>0.486714</td>
      <td>4.854853</td>
      <td>548.021466</td>
      <td>2.161418</td>
      <td>1.386694</td>
      <td>1.811395</td>
    </tr>
    <tr>
      <th>min</th>
      <td>579533.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>596484.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>14.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>615479.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>248.000000</td>
      <td>2.000000</td>
      <td>14.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>637387.250000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>936.000000</td>
      <td>3.000000</td>
      <td>15.000000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>660737.000000</td>
      <td>1.000000</td>
      <td>251.000000</td>
      <td>2272.000000</td>
      <td>55.000000</td>
      <td>18.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# to know how many null values are present in every columns
df.isnull().sum()
```




    Prospect ID                                         0
    Lead Number                                         0
    Lead Origin                                         0
    Lead Source                                        36
    Do Not Email                                        0
    Do Not Call                                         0
    Converted                                           0
    TotalVisits                                       137
    Total Time Spent on Website                         0
    Page Views Per Visit                              137
    Last Activity                                     103
    Country                                          2461
    Specialization                                   1438
    How did you hear about X Education               2207
    What is your current occupation                  2690
    What matters most to you in choosing a course    2709
    Search                                              0
    Magazine                                            0
    Newspaper Article                                   0
    X Education Forums                                  0
    Newspaper                                           0
    Digital Advertisement                               0
    Through Recommendations                             0
    Receive More Updates About Our Courses              0
    Tags                                             3353
    Lead Quality                                     4767
    Update me on Supply Chain Content                   0
    Get updates on DM Content                           0
    Lead Profile                                     2709
    City                                             1420
    Asymmetrique Activity Index                      4218
    Asymmetrique Profile Index                       4218
    Asymmetrique Activity Score                      4218
    Asymmetrique Profile Score                       4218
    I agree to pay the amount through cheque            0
    A free copy of Mastering The Interview              0
    Last Notable Activity                               0
    dtype: int64




```python
# to know how many null values are present in every columns in terms of percentage
round(df.isnull().sum()/len(df)*100,2)
```




    Prospect ID                                       0.00
    Lead Number                                       0.00
    Lead Origin                                       0.00
    Lead Source                                       0.39
    Do Not Email                                      0.00
    Do Not Call                                       0.00
    Converted                                         0.00
    TotalVisits                                       1.48
    Total Time Spent on Website                       0.00
    Page Views Per Visit                              1.48
    Last Activity                                     1.11
    Country                                          26.63
    Specialization                                   15.56
    How did you hear about X Education               23.89
    What is your current occupation                  29.11
    What matters most to you in choosing a course    29.32
    Search                                            0.00
    Magazine                                          0.00
    Newspaper Article                                 0.00
    X Education Forums                                0.00
    Newspaper                                         0.00
    Digital Advertisement                             0.00
    Through Recommendations                           0.00
    Receive More Updates About Our Courses            0.00
    Tags                                             36.29
    Lead Quality                                     51.59
    Update me on Supply Chain Content                 0.00
    Get updates on DM Content                         0.00
    Lead Profile                                     29.32
    City                                             15.37
    Asymmetrique Activity Index                      45.65
    Asymmetrique Profile Index                       45.65
    Asymmetrique Activity Score                      45.65
    Asymmetrique Profile Score                       45.65
    I agree to pay the amount through cheque          0.00
    A free copy of Mastering The Interview            0.00
    Last Notable Activity                             0.00
    dtype: float64



##### **`As we know we can't create a model with null values we need to remove all null values from the dataset`**
##### **`let's start inspecting each column one by one with containing null values`**


```python
df['City'].value_counts()
```




    City
    Mumbai                         3222
    Select                         2249
    Thane & Outskirts               752
    Other Cities                    686
    Other Cities of Maharashtra     457
    Other Metro Cities              380
    Tier II Cities                   74
    Name: count, dtype: int64




```python
# while inspection I found 'Select' is the common value in most of the column thus it is good to replace all null values for every column with 'Select'
```


```python
df['Country'].replace(np.nan,'Select', inplace=True)
```


```python
df['Specialization'].replace(np.nan,'Select', inplace=True)
```


```python
df['How did you hear about X Education'].replace(np.nan,'Select', inplace=True)
```


```python
df['What is your current occupation'].replace(np.nan,'Select', inplace=True)
```


```python
df['What matters most to you in choosing a course'].replace(np.nan,'Select', inplace=True)
```


```python
df['Tags'].replace(np.nan,'No Comments', inplace=True)
```


```python
df['Lead Quality'].replace(np.nan,'Not Sure', inplace=True)
```


```python
df['Lead Profile'].replace(np.nan,'Select', inplace=True)
```


```python
df['City'].replace(np.nan,'Select', inplace=True)
```


```python
df.isnull().sum()
```




    Prospect ID                                         0
    Lead Number                                         0
    Lead Origin                                         0
    Lead Source                                        36
    Do Not Email                                        0
    Do Not Call                                         0
    Converted                                           0
    TotalVisits                                       137
    Total Time Spent on Website                         0
    Page Views Per Visit                              137
    Last Activity                                     103
    Country                                             0
    Specialization                                      0
    How did you hear about X Education                  0
    What is your current occupation                     0
    What matters most to you in choosing a course       0
    Search                                              0
    Magazine                                            0
    Newspaper Article                                   0
    X Education Forums                                  0
    Newspaper                                           0
    Digital Advertisement                               0
    Through Recommendations                             0
    Receive More Updates About Our Courses              0
    Tags                                                0
    Lead Quality                                        0
    Update me on Supply Chain Content                   0
    Get updates on DM Content                           0
    Lead Profile                                        0
    City                                                0
    Asymmetrique Activity Index                      4218
    Asymmetrique Profile Index                       4218
    Asymmetrique Activity Score                      4218
    Asymmetrique Profile Score                       4218
    I agree to pay the amount through cheque            0
    A free copy of Mastering The Interview              0
    Last Notable Activity                               0
    dtype: int64




```python
# here we don't need these columns which contains null value so we can drop them

df.drop(['Asymmetrique Activity Index',
       'Asymmetrique Profile Index', 'Asymmetrique Activity Score',
       'Asymmetrique Profile Score'], axis=1, inplace=True)
```

**As we know in these column there are very less amount of null values so we can drop that rows which contains null values**
Lead Source                                        36
TotalVisits                                       137
Page Views Per Visit                              137
Last Activity                                     103
**total 413 rows will be deleted from 9240 rows (9240-413 = 8827)**


```python
# this method will drop rows which contains null values
df.dropna(inplace=True)
```


```python
# to know how many null values are present in every columns in terms of percentage, after handling null values
round(df.isnull().sum()/len(df)*100,2)
```




    Prospect ID                                      0.0
    Lead Number                                      0.0
    Lead Origin                                      0.0
    Lead Source                                      0.0
    Do Not Email                                     0.0
    Do Not Call                                      0.0
    Converted                                        0.0
    TotalVisits                                      0.0
    Total Time Spent on Website                      0.0
    Page Views Per Visit                             0.0
    Last Activity                                    0.0
    Country                                          0.0
    Specialization                                   0.0
    How did you hear about X Education               0.0
    What is your current occupation                  0.0
    What matters most to you in choosing a course    0.0
    Search                                           0.0
    Magazine                                         0.0
    Newspaper Article                                0.0
    X Education Forums                               0.0
    Newspaper                                        0.0
    Digital Advertisement                            0.0
    Through Recommendations                          0.0
    Receive More Updates About Our Courses           0.0
    Tags                                             0.0
    Lead Quality                                     0.0
    Update me on Supply Chain Content                0.0
    Get updates on DM Content                        0.0
    Lead Profile                                     0.0
    City                                             0.0
    I agree to pay the amount through cheque         0.0
    A free copy of Mastering The Interview           0.0
    Last Notable Activity                            0.0
    dtype: float64



### Encoding

##### `converting all object columns into numerical columns`


```python
# As we know datatypes for all column should be a number for creating a AI Model
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 9074 entries, 0 to 9239
    Data columns (total 33 columns):
     #   Column                                         Non-Null Count  Dtype  
    ---  ------                                         --------------  -----  
     0   Prospect ID                                    9074 non-null   object 
     1   Lead Number                                    9074 non-null   int64  
     2   Lead Origin                                    9074 non-null   object 
     3   Lead Source                                    9074 non-null   object 
     4   Do Not Email                                   9074 non-null   object 
     5   Do Not Call                                    9074 non-null   object 
     6   Converted                                      9074 non-null   int64  
     7   TotalVisits                                    9074 non-null   float64
     8   Total Time Spent on Website                    9074 non-null   int64  
     9   Page Views Per Visit                           9074 non-null   float64
     10  Last Activity                                  9074 non-null   object 
     11  Country                                        9074 non-null   object 
     12  Specialization                                 9074 non-null   object 
     13  How did you hear about X Education             9074 non-null   object 
     14  What is your current occupation                9074 non-null   object 
     15  What matters most to you in choosing a course  9074 non-null   object 
     16  Search                                         9074 non-null   object 
     17  Magazine                                       9074 non-null   object 
     18  Newspaper Article                              9074 non-null   object 
     19  X Education Forums                             9074 non-null   object 
     20  Newspaper                                      9074 non-null   object 
     21  Digital Advertisement                          9074 non-null   object 
     22  Through Recommendations                        9074 non-null   object 
     23  Receive More Updates About Our Courses         9074 non-null   object 
     24  Tags                                           9074 non-null   object 
     25  Lead Quality                                   9074 non-null   object 
     26  Update me on Supply Chain Content              9074 non-null   object 
     27  Get updates on DM Content                      9074 non-null   object 
     28  Lead Profile                                   9074 non-null   object 
     29  City                                           9074 non-null   object 
     30  I agree to pay the amount through cheque       9074 non-null   object 
     31  A free copy of Mastering The Interview         9074 non-null   object 
     32  Last Notable Activity                          9074 non-null   object 
    dtypes: float64(2), int64(3), object(28)
    memory usage: 2.4+ MB
    


```python
# to know all (object and numerical both) distinct value counts of each columns
for cols in df.columns:
    print(cols, ":", df[cols].nunique())
```

    Prospect ID : 9074
    Lead Number : 9074
    Lead Origin : 4
    Lead Source : 21
    Do Not Email : 2
    Do Not Call : 2
    Converted : 2
    TotalVisits : 41
    Total Time Spent on Website : 1717
    Page Views Per Visit : 114
    Last Activity : 17
    Country : 39
    Specialization : 19
    How did you hear about X Education : 10
    What is your current occupation : 7
    What matters most to you in choosing a course : 4
    Search : 2
    Magazine : 1
    Newspaper Article : 2
    X Education Forums : 2
    Newspaper : 2
    Digital Advertisement : 2
    Through Recommendations : 2
    Receive More Updates About Our Courses : 1
    Tags : 27
    Lead Quality : 5
    Update me on Supply Chain Content : 1
    Get updates on DM Content : 1
    Lead Profile : 6
    City : 7
    I agree to pay the amount through cheque : 1
    A free copy of Mastering The Interview : 2
    Last Notable Activity : 16
    


```python
# to know distinct value counts of all object columns
# here we can find that which type of encoding we've to be perform in which object column

for cols in df.columns:
    if df[cols].dtype == 'object':
        print(cols, ":", df[cols].nunique())
```

    Prospect ID : 9074
    Lead Origin : 4
    Lead Source : 21
    Do Not Email : 2
    Do Not Call : 2
    Last Activity : 17
    Country : 39
    Specialization : 19
    How did you hear about X Education : 10
    What is your current occupation : 7
    What matters most to you in choosing a course : 4
    Search : 2
    Magazine : 1
    Newspaper Article : 2
    X Education Forums : 2
    Newspaper : 2
    Digital Advertisement : 2
    Through Recommendations : 2
    Receive More Updates About Our Courses : 1
    Tags : 27
    Lead Quality : 5
    Update me on Supply Chain Content : 1
    Get updates on DM Content : 1
    Lead Profile : 6
    City : 7
    I agree to pay the amount through cheque : 1
    A free copy of Mastering The Interview : 2
    Last Notable Activity : 16
    

#### **`distinct value of 2 are good contender for binary encoding (0 and 1)`**
#### **`distinct value of <10 are good contender for one-hot encoding or dummy variable creation`**
#### **`distinct value of >10 are good contender for frequency encoding`**


```python
# let's start inspecting all elegible columns for binary encoding
```


```python
df['Newspaper'].value_counts()
```




    Newspaper
    No     9073
    Yes       1
    Name: count, dtype: int64




```python
# df['Do Not Email'].replace({'No':0,'Yes':1}, inplace=True)   #Replacing 'No' with 0 and 'Yes' with 1
```


```python
# df['Do Not Call'].replace({'No':0,'Yes':1}, inplace=True)   #Replacing 'No' with 0 and 'Yes' with 1
```


```python
# df['Search'].replace({'No':0,'Yes':1}, inplace=True)   #Replacing 'No' with 0 and 'Yes' with 1
```


```python
# df['Newspaper Article'].replace({'No':0,'Yes':1}, inplace=True)   #Replacing 'No' with 0 and 'Yes' with 1
```


```python
# df['X Education Forums'].replace({'No':0,'Yes':1}, inplace=True)   #Replacing 'No' with 0 and 'Yes' with 1
```


```python
# df['Newspaper'].replace({'No':0,'Yes':1}, inplace=True)   #Replacing 'No' with 0 and 'Yes' with 1
```


```python
# df['Digital Advertisement'].replace({'No':0,'Yes':1}, inplace=True)   #Replacing 'No' with 0 and 'Yes' with 1
```


```python
# df['Through Recommendations'].replace({'No':0,'Yes':1}, inplace=True)   #Replacing 'No' with 0 and 'Yes' with 1
```


```python
# df['A free copy of Mastering The Interview'].replace({'No':0,'Yes':1}, inplace=True)   #Replacing 'No' with 0 and 'Yes' with 1
```


```python
# Replacing 'No' with 0 and 'Yes' with 1

bin = ['Do Not Email', 'Do Not Call', 'Search', 'Newspaper Article', 'X Education Forums','Newspaper','Digital Advertisement',
       'Through Recommendations', 'A free copy of Mastering The Interview']

# creating a function for binary mapping
def binary_map(x):
    return x.map({'No':0,'Yes':1})

# Applying the function to the bin list
df[bin] = df[bin].apply(binary_map)
```


```python
for cols in df.columns:
    if df[cols].dtype == 'object':
        print(cols, ":", df[cols].nunique())

# distinct values of 2 has gone now
```

    Prospect ID : 9074
    Lead Origin : 4
    Lead Source : 21
    Last Activity : 17
    Country : 39
    Specialization : 19
    How did you hear about X Education : 10
    What is your current occupation : 7
    What matters most to you in choosing a course : 4
    Magazine : 1
    Receive More Updates About Our Courses : 1
    Tags : 27
    Lead Quality : 5
    Update me on Supply Chain Content : 1
    Get updates on DM Content : 1
    Lead Profile : 6
    City : 7
    I agree to pay the amount through cheque : 1
    Last Notable Activity : 16
    


```python
# As we know that count of distinct values less than 10 are applicable for dummy variable creation so now we need to encode these columns

dum = pd.get_dummies(df[['Lead Origin','What is your current occupation','What matters most to you in choosing a course',
                         'Lead Quality','Lead Profile','City']], drop_first=True, dtype=int)

# 4-1=3
# 7-1=6
# 4-1=3
# 5-1=4
# 6-1=5
# 7-1=6
# 27 new columns will created
```


```python
dum
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lead Origin_Landing Page Submission</th>
      <th>Lead Origin_Lead Add Form</th>
      <th>Lead Origin_Lead Import</th>
      <th>What is your current occupation_Housewife</th>
      <th>What is your current occupation_Other</th>
      <th>What is your current occupation_Select</th>
      <th>What is your current occupation_Student</th>
      <th>What is your current occupation_Unemployed</th>
      <th>What is your current occupation_Working Professional</th>
      <th>What matters most to you in choosing a course_Flexibility &amp; Convenience</th>
      <th>...</th>
      <th>Lead Profile_Other Leads</th>
      <th>Lead Profile_Potential Lead</th>
      <th>Lead Profile_Select</th>
      <th>Lead Profile_Student of SomeSchool</th>
      <th>City_Other Cities</th>
      <th>City_Other Cities of Maharashtra</th>
      <th>City_Other Metro Cities</th>
      <th>City_Select</th>
      <th>City_Thane &amp; Outskirts</th>
      <th>City_Tier II Cities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9235</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9236</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9237</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9238</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9239</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>9074 rows × 27 columns</p>
</div>




```python
# now concat df and newly created columns named dum to a new dataframe and call this newdf
newdf = pd.concat([df, dum], axis=1)
```


```python
newdf.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prospect ID</th>
      <th>Lead Number</th>
      <th>Lead Origin</th>
      <th>Lead Source</th>
      <th>Do Not Email</th>
      <th>Do Not Call</th>
      <th>Converted</th>
      <th>TotalVisits</th>
      <th>Total Time Spent on Website</th>
      <th>Page Views Per Visit</th>
      <th>...</th>
      <th>Lead Profile_Other Leads</th>
      <th>Lead Profile_Potential Lead</th>
      <th>Lead Profile_Select</th>
      <th>Lead Profile_Student of SomeSchool</th>
      <th>City_Other Cities</th>
      <th>City_Other Cities of Maharashtra</th>
      <th>City_Other Metro Cities</th>
      <th>City_Select</th>
      <th>City_Thane &amp; Outskirts</th>
      <th>City_Tier II Cities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7927b2df-8bba-4d29-b9a2-b6e0beafe620</td>
      <td>660737</td>
      <td>API</td>
      <td>Olark Chat</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2a272436-5132-4136-86fa-dcc88c88f482</td>
      <td>660728</td>
      <td>API</td>
      <td>Organic Search</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>674</td>
      <td>2.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8cc8c611-a219-4f35-ad23-fdfd2656bd8a</td>
      <td>660727</td>
      <td>Landing Page Submission</td>
      <td>Direct Traffic</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1532</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0cc2df48-7cf4-4e39-9de9-19797f9b38cc</td>
      <td>660719</td>
      <td>Landing Page Submission</td>
      <td>Direct Traffic</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>305</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3256f628-e534-4826-9d63-4a8b88782852</td>
      <td>660681</td>
      <td>Landing Page Submission</td>
      <td>Google</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1428</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>




```python
# now drop these columns because we already created dummy valiables for these columns so we don't need this

newdf.drop(['Lead Origin','What is your current occupation','What matters most to you in choosing a course',
                         'Lead Quality','Lead Profile','City'],axis=1, inplace=True)
```


```python
for cols in newdf.columns:
    if newdf[cols].dtype == 'object':
        print(cols, ":", newdf[cols].nunique())

# columns for the dummy valiable creation are also missing
```

    Prospect ID : 9074
    Lead Source : 21
    Last Activity : 17
    Country : 39
    Specialization : 19
    How did you hear about X Education : 10
    Magazine : 1
    Receive More Updates About Our Courses : 1
    Tags : 27
    Update me on Supply Chain Content : 1
    Get updates on DM Content : 1
    I agree to pay the amount through cheque : 1
    Last Notable Activity : 16
    


```python
# here we don't need that columns which has only 1 distinct value so we need to drop those

newdf.drop(['Magazine','X Education Forums','Newspaper','Receive More Updates About Our Courses','Update me on Supply Chain Content',
            'Get updates on DM Content','I agree to pay the amount through cheque'],axis=1, inplace=True)
```


```python
for cols in newdf.columns:
    if newdf[cols].dtype == 'object':
        print(cols, ":", newdf[cols].nunique())

# columns for only one unique value are also missing now
```

    Prospect ID : 9074
    Lead Source : 21
    Last Activity : 17
    Country : 39
    Specialization : 19
    How did you hear about X Education : 10
    Tags : 27
    Last Notable Activity : 16
    


```python
# here Prospect ID isn't giving any valuable information so we can drop it too
newdf.drop('Prospect ID',axis=1, inplace=True)
```


```python
for cols in newdf.columns:
    if newdf[cols].dtype == 'object':
        print(cols, ":", newdf[cols].nunique())

# missing Prospect ID
```

    Lead Source : 21
    Last Activity : 17
    Country : 39
    Specialization : 19
    How did you hear about X Education : 10
    Tags : 27
    Last Notable Activity : 16
    


```python
# Lead Source : 18
# Last Activity : 16
# Country : 34
# Specialization : 19
# How did you hear about X Education : 10
# Tags : 26
# Last Notable Activity : 13

# all these columns are applicable for frequency encoding


```


```python
newdf['Lead Source'] = newdf['Lead Source'].map(newdf['Lead Source'].value_counts())
```


```python
newdf['Last Activity'] = newdf['Last Activity'].map(newdf['Last Activity'].value_counts())
```


```python
newdf['Country'] = newdf['Country'].map(newdf['Country'].value_counts())
```


```python
newdf['Specialization'] = newdf['Specialization'].map(newdf['Specialization'].value_counts())
```


```python
newdf['How did you hear about X Education'] = newdf['How did you hear about X Education'].map(newdf['How did you hear about X Education'].value_counts())
```


```python
newdf['Tags'] = newdf['Tags'].map(newdf['Tags'].value_counts())
```


```python
newdf['Last Notable Activity'] = newdf['Last Notable Activity'].map(newdf['Last Notable Activity'].value_counts())
```


```python
for cols in newdf.columns:
    if newdf[cols].dtype == 'object':
        print(cols, ":", newdf[cols].nunique())

# done all frequency encoding
```


```python
newdf.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lead Number</th>
      <th>Lead Source</th>
      <th>Do Not Email</th>
      <th>Do Not Call</th>
      <th>Converted</th>
      <th>TotalVisits</th>
      <th>Total Time Spent on Website</th>
      <th>Page Views Per Visit</th>
      <th>Last Activity</th>
      <th>Country</th>
      <th>...</th>
      <th>Lead Profile_Other Leads</th>
      <th>Lead Profile_Potential Lead</th>
      <th>Lead Profile_Select</th>
      <th>Lead Profile_Student of SomeSchool</th>
      <th>City_Other Cities</th>
      <th>City_Other Cities of Maharashtra</th>
      <th>City_Other Metro Cities</th>
      <th>City_Select</th>
      <th>City_Thane &amp; Outskirts</th>
      <th>City_Tier II Cities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>660737</td>
      <td>1753</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>640</td>
      <td>2296</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>660728</td>
      <td>1154</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>674</td>
      <td>2.5</td>
      <td>3432</td>
      <td>6491</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>660727</td>
      <td>2543</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1532</td>
      <td>2.0</td>
      <td>3432</td>
      <td>6491</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>660719</td>
      <td>2543</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>305</td>
      <td>1.0</td>
      <td>90</td>
      <td>6491</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>660681</td>
      <td>2868</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1428</td>
      <td>1.0</td>
      <td>428</td>
      <td>6491</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>




```python
# I think 'lead number' and 'lead source' isn't giving any valuable info so we can also drop them

newdf.drop(['Lead Number', 'Lead Source'], axis=1, inplace=True)
```


```python
newdf.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Do Not Email</th>
      <th>Do Not Call</th>
      <th>Converted</th>
      <th>TotalVisits</th>
      <th>Total Time Spent on Website</th>
      <th>Page Views Per Visit</th>
      <th>Last Activity</th>
      <th>Country</th>
      <th>Specialization</th>
      <th>How did you hear about X Education</th>
      <th>...</th>
      <th>Lead Profile_Other Leads</th>
      <th>Lead Profile_Potential Lead</th>
      <th>Lead Profile_Select</th>
      <th>Lead Profile_Student of SomeSchool</th>
      <th>City_Other Cities</th>
      <th>City_Other Cities of Maharashtra</th>
      <th>City_Other Metro Cities</th>
      <th>City_Select</th>
      <th>City_Thane &amp; Outskirts</th>
      <th>City_Tier II Cities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>640</td>
      <td>2296</td>
      <td>3282</td>
      <td>7086</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>674</td>
      <td>2.5</td>
      <td>3432</td>
      <td>6491</td>
      <td>3282</td>
      <td>7086</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1532</td>
      <td>2.0</td>
      <td>3432</td>
      <td>6491</td>
      <td>399</td>
      <td>7086</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>305</td>
      <td>1.0</td>
      <td>90</td>
      <td>6491</td>
      <td>202</td>
      <td>347</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1428</td>
      <td>1.0</td>
      <td>428</td>
      <td>6491</td>
      <td>3282</td>
      <td>186</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 44 columns</p>
</div>




```python

```
