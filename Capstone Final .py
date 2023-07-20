#!/usr/bin/env python
# coding: utf-8

# <h1><center>Data Analytics Graduate Capstone - D214</center></h1>
# <h1><center>Instructor: Daniel Smith</center></h1>
# <h1><center>Master of Science, Data Analytics Program</center></h1>
# <h1><center>Data Analytics Graduate Capstone</center></h1>
# <h1><center>Brittany Abney</center></h1>
# <h1><center>Capstone Project: Data Science Salaries & Trends</center></h1>
# <h1><center>July 16th, 2023</center></h1>
# <h1><center>Expected Completion Date: August 1st, 2023</center></h1>

# ###  A - Research Question:
# 
# 
# 
# #### Context: 
# 
# The most common reason for employee retention problems is salary dissatisfaction. (3. n.D.,2023) Providing employers with the ability to be competitive in their offerings to other companies can help increase retention and save money. 
# 
# Glassdoor is a company that is not only useful to employees, but also to employers just the same, if not more. The best way for a company to save money is through retainment. If an employee is happy with where they work and comfortable with their salary, they are less likely to leave. This allows companies to spend less time rehiring, onboarding, and training.
# 
# Glassdoor provides employees with the opportunity to hear about a company from employees who work there. It helps to decide if that is the company that will make them happy to work at. This is a major benefit to the employer, as employee sentiment is easily analyzed when employees provide feedback. Glassdoor not only allows employers to analyze employee feedback, but it also provides a multitude of salary information based on job titles. Employers can utilize this data to see how they compare to other companies and if there is any room for improvement or savings. (1. Raelson, 2018) Employers can also predict, based on prior salary data, what range they should be in for the upcoming fiscal year.
# 
# Apollo Technical released an article stating, "It costs an employer an average of 33% of an employee's yearly salary for their exit." Companies should follow the general rule of obtaining and maintaining a 90% or higher retention rate. (2. Apollo Technical, 2023) The most common reasons for low retention rates are a lack of training, poor communication, a lack of recognition, and a low salary.
# 
# 
# #### Research Question: 
# To what extent do the statistically significant variables of company size, company rating, and job title affect and predict data science salaries?
# 
# Justification: Employers should know the average salary for a role and have comparisons with their competitors. Analyzing this data will allow us to predict data science salaries employers can use in 2025 and help businesses identify if they are at appropriate pay scales for their talent range.
# 
# 
# #### Hypothesis: 
# 
# Variables of company size, company rating, and job title have statistically significant impacts on data science salaries and predictions.
# 
# #### Discussion of hypothesis: 
# 
# The data I am using will allow me to compare salaries by company size, company rating, and job title. 
# 
# #### Project Outcomes: 
# 
# The anticipated project outcomes are to identify the statistical significance of the three variables: company size, company rating, and job title. I will utilize different analysis methods to provide actionable insight into data science salaries and comparisons. I will provide insight into what 2025 salary predictions should be. I will also help identify the highest-ranked competitors for talented data science candidates and the ratings of those competitors. I will provide visual aids that help to represent the results of the analysis. 

# ## B: Data

# #### Data Collection:
# 
# I chose to use Kaggle to obtain my data. I downloaded three CSV files from three different Kaggle users. I selected two reports that were scraped from Glassdoor for different timeframes. The first dataset is Glassdoor salaries and data science job listing details that include: job title, salary estimate, job description, rating, company, location, headquarter location, size of company, rating, industry, sector, revenue, and competitors. This data was scraped from Glassdoor by Hidayat Zeb a month ago and shows current data science salaries. This dataset has 743 entries. My second dataset is also from Kaggle and was web scraped by Kaggle user Larxel. This dataset states it was pulled amidst a pandemic and appears to have been last updated three years ago. As the COVID pandemic was in 2021, this leads me to believe this was pulled in late 2021. There are 2253 entries.
# 
# The last dataset does not match the first two except by company name. Joy Shil, the user who created this dataset, is a teacher assistant in Bangladesh and a Kaggle expert. The data was web scraped from Glassdoor on the USA’s top companies and provides: (9940 entries) different company ratings, total amount of salaries listed, total number of company jobs, location, number of employees, industry type, and company description.
# 
# 1:	https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023/discussion/414036  (Zeb, 2023)
# 
# 2:	https://www.kaggle.com/datasets/andrewmvd/data-analyst-jobs (Larxel, 2020)
# 
# 3:	USA Company Insights: Glassdoor Scraped Data 2023 | Kaggle (Shil, 2023)
# 
# 
# #### One advantage of the data-gathering methodology I used:
# 
# Since I chose to use prescraped data, this saved me significant time by not having to scrape Glassdoor for the information I needed.
# 
# #### One disadvantage of the data-gathering methodology I used:
# 
# The third dataset did not match the first two as it only provided company review information. I needed to find a way to combine the three to provide an accurate analysis. Upon cleaning the data and multiple attempts, I decided the best way to utilize this data was to only include employers listed in the first two datasets. I combined the data by company name and removed any extra company information that was not needed.
# 
# #### Permissions to use the data:
# 
# Glassdoor owns the data but clearly states on their site that all publicly accessible data is free to use. All the data scraped is available to the public. I also attached an email from Glassdoor stating that their public data is free to use.
# 
# 
# 

# # C - Data Extraction & Preparation

# #### Data Extraction/Tools & Techniques:
# The Kaggle site allowed me to utilize CSV files that were pre-scraped from Glassdoor. I loaded each dataset into its own dataframe. I did minor cleaning on the first two datasets and then combined the two. I then did a more thorough cleaning. After that, I cleaned the third dataset and used the concat merge feature to combine it by company name. Please see each step's description below.
# 
# #### Justification:
# Utilizing libaries such as Pandas, Matplotlib, Numpy, and SciPy is the fastest way for me to get the basic parts of the data clean, identify null values that may need to be removed, and prepare the data for linear regression modeling. The cleaning process also helps to get comfortable with the data and understand it. 
# 
# The scraped datasets had many columns filled with a mix of integers and characters. This was causing datatypes for objects. This had to be fixed. I would say this was the most difficult part of cleaning and preparing the data. Str.replace was utilized quite a bit due to strings being imputable. Instead of modifying the string, this function returns a copy of the object where x is replaced with " ". I ran into many roadblocks as some of the string data was fighting to be made into a float, but I was able to use the replace function to remove the data that was causing issues. Another important piece was the Python string split method. The split method splits a string into a list. Both of these functions allowed me to clean the data to a point where it would be model-ready.
# 
# The main disadvantage of downloading CSV files instead of scraping is that I had to clean up data that was not obtained in the format I would have preferred. Both the split function and str.replace allowed me to bypass this disadvantage.
# 
# The one advantage of utilizing the downloaded datasets was that it meant less time scraping and more time to clean and analyze!

# #### Below, I wil walk you through loading my data and the cleaning process to prepare it for analysis.
# 
# There are so mnay libraries that can be useful in Python. I chose to utilize so many different and really utilize Python's capabilities.
# 

# In[1]:


#Load libaries
# Import neccessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import ydata_profiling
from scipy import stats
import statsmodels.api as sm

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.style.use('ggplot')

# Scikit-learn
import sklearn as sl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from scipy.stats import linregress
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from statsmodels.formula.api import ols
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D  
get_ipython().run_line_magic('matplotlib', 'inline')
ols = linear_model.LinearRegression()

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingClassifier
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RepeatedKFold

import graphviz
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree

# hide warnings
import warnings
warnings.filterwarnings('ignore')

import chardet
import nltk


# Since two out of three datasets were very similar.I chose to load these two first and combine. Once I had one full clean combined set, I could load the third dataset and combine it. This allowed me to really get a feel for each dataset indivdiually.

# In[2]:


#Load my first two datasets and add column to 1 and 2 with year of data.
#load the dataset
companydata1 = pd.read_csv("Dataset 1.csv")
companydata1['Year'] = '2023'

#load the dataset
companydata2 = pd.read_csv("Dataset 2.csv")
companydata2['Year'] = '2021'

#Check shape
companydata1.shape,companydata2.shape


# In[3]:


#Rename columns
companydata1 = companydata1.rename(columns={'Job Title': 'jobtitle', 'Salary Estimate': 'salary_estimate', 'Rating':'rating', 'Company Name': 'companyname', 'Location':'location', 'Size': 'size', 'Industry':'industry', 'Competitors':'competitors'})
companydata2 = companydata2.rename(columns={'Job Title': 'jobtitle', 'Salary Estimate': 'salary_estimate', 'Rating':'rating', 'Company Name': 'companyname', 'Location':'location', 'Size': 'size', 'Industry':'industry', 'Competitors':'competitors'})

#Remove columns that are not needed
#Remove the columns I do not need.
companydata1 = companydata1.drop(['Founded', 'min_salary', 'max_salary', 'avg_salary', 'employer_provided', 'company_txt', 'same_state', 'age','python_yn', 'R_yn', 'spark', 'aws', 'excel', 'hourly', 'Job Description', 'Headquarters', 'Type of ownership', 'Sector', 'job_state', 'Revenue', 'industry'], axis = 1)
companydata2 = companydata2.drop(['Founded', 'Unnamed: 0', 'Job Description', 'Headquarters', 'Type of ownership', 'Sector', 'Revenue', 'Easy Apply', 'industry'], axis = 1)

#Check shape again 
companydata1.shape, companydata2.shape


# In[4]:


#Combine dataset 1 and 2 for cleaning. 
# (C1. Chen, 2020)

jobsdf = pd.concat([companydata1, companydata2], ignore_index=True)
jobsdf.info()


# In[5]:


#We will not need the competitors column for this analysis nor will we need location.
jobsdf = jobsdf.drop(['competitors', 'location'], axis = 1)

#Check for null values
jobsdf.isna().any()

#Drop null values
jobsdf = jobsdf.dropna(how='any',axis=0) 

#Shape check 
jobsdf.shape


# The initial dataset had 742 rows with 29 columns and the second dataset had 2253 rows with 17 columns. Combining and minimal cleaning keaves us with 2994 rows and 6 columns.
# I need to visulize outliers before continuing with cleaning.

# In[6]:


#Check out the data before it's clean for outliers.
jobsDFplot = pd.DataFrame(np.random.randn(10, 4),
                  columns=['jobtitle', 'salary_estimate', 'rating', 'size'])
boxplot = jobsDFplot.boxplot(column=['jobtitle', 'salary_estimate', 'rating', 'size'])


# We can see that job title has an outlier and so does rating and size.

# In[7]:


#CLean the salary column
jobsdf['min_sal'] = jobsdf['salary_estimate'].str.split(",").str[0].str.replace('(Glassdoor est.)','')
jobsdf['min_sal'] = jobsdf['min_sal'].str.split(",").str[0].str.replace('(Glassdoor est.)','')
jobsdf['min_sal'] = jobsdf['min_sal'].str.replace('(Glassdoor est.)','').str.split('-').str[0].str.replace('$','').str.replace('K','')
jobsdf['min_sal'] = jobsdf['min_sal'].str.replace('(Glassdoor est.)','')
jobsdf['min_sal'] = jobsdf['min_sal'].str.replace('Employer Provided Salary:', '' )
jobsdf['min_sal'] = jobsdf['min_sal'].str.replace('!', '' )

jobsdf['max_sal'] = jobsdf['salary_estimate'].str.split(",").str[0].str.replace('(Glassdoor est.)','')
jobsdf['max_sal'] = jobsdf['max_sal'].str.replace('(Glassdoor est.)','').str.split('-').str[1].str.replace('$','').str.replace('K','')
jobsdf['max_sal'] = jobsdf['max_sal'].str.replace('(Employer est.)','')
jobsdf['max_sal'] = jobsdf['max_sal'].str.split().str[0].str.replace('(','').str.replace(')','')
jobsdf['max_sal'].unique()

jobsdf['min_sal'] = jobsdf['min_sal'].replace(np.nan, jobsdf['min_sal'].mean())
jobsdf['max_sal'] = jobsdf['max_sal'].replace(np.nan, jobsdf['max_sal'].mean())

jobsdf['min_sal'] = pd.to_numeric(jobsdf['min_sal'], errors='coerce')
type(jobsdf['min_sal'])

jobsdf['max_sal'] = pd.to_numeric(jobsdf['max_sal'], errors='coerce')
type(jobsdf['max_sal'])

jobsdf['avg_salary'] = (jobsdf['min_sal'] + jobsdf['max_sal'])/ 2

#Check for new null values
jobsdf['min_sal'].isna().sum(), jobsdf['max_sal'].isna().sum()
#Remove 1 naan value and recheck.
jobsdf['min_sal'] = jobsdf['min_sal'].replace(np.nan, jobsdf['min_sal'].mean())
jobsdf['max_sal'] = jobsdf['max_sal'].replace(np.nan, jobsdf['max_sal'].mean())

jobsdf['min_sal'].isna().sum()
#Remove salary_estimate column as it has been replaced with the minimum, maximum, and average.
jobsdf = jobsdf.drop(['salary_estimate'], axis = 1)
jobsdf.head()


# In[8]:


#Clean jobtitle column
def jobtitle_cleaner(jobtitle):
    if 'scientist' in jobtitle.lower():
        return 'scientist'
    elif 'engineer' in jobtitle.lower():
        return 'engineer'
    elif 'analyst' in jobtitle.lower():
        return 'analyst'
    elif 'senior' or 'principal' in jobtitle.lower():
        return 'senior roles'
    else:
        return ''
    
jobsdf['jobtitleClean'] = jobsdf['jobtitle'].apply(jobtitle_cleaner)

#Check for null values
jobsdf['jobtitleClean'].isnull()

jobsdf = jobsdf.dropna(how='any',axis=0) 
#Remove the no longer needed job title column
jobsdf = jobsdf.drop(['jobtitle'], axis = 1)
jobsdf.head()


# In[9]:


#Clean company name which will only be used for visualizing before modeling.
jobsdf['companyname'][0].split('\n')[0]
# remove numerical features from company name
jobsdf['companyname'] = jobsdf['companyname'].apply(lambda x : x.split("\n")[0])
jobsdf['companyname']

#Clean company size column

jobsdf['size'] = jobsdf['size'].str.replace('employees','').str.replace('to', '').str.replace('+', '').str.replace('Unknown', '1').str.replace('-', '')
jobsdf['size'] = jobsdf['size'].str.replace('501  1000', '4').str.replace('10000 ', '7').str.replace('1001  5000 ','5').str.replace('51  200 ', '2').str.replace('201  500 ', '3').str.replace('5001  10000 ', '6').str.replace('1  50 ', '1').str.replace('1', '1')
jobsdf['size'] = jobsdf['size'].str.replace('5001  7', '6')
jobsdf['size'] = jobsdf['size'].str.replace('\'', '')

#Check df datayptes.
jobsdf.dtypes

pd.to_numeric(jobsdf['size'], errors='coerce').notnull().all()

jobsdf['size'] = jobsdf['size'].astype(str).astype(int)
jobsdf['Year'] = pd.to_numeric(jobsdf['Year'])


print (jobsdf.dtypes)


# In[10]:


#remove rows that are duplicated
jobsdf.drop_duplicates(keep=False, inplace=False)

#Review cleaned and combined two datasets
jobsdf.head()


# In[11]:


#Load the third dataframe and check shape.
with open('Dataset 3.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large


companydata3 = pd.read_csv('Dataset 3.csv', encoding=result['encoding'])

companydata3.shape


# In[12]:


companydata3.head(5)


# In[13]:


#Drop the columns I do not want to visualize

companydata3 = companydata3.drop(['Company Jobs', 'Company reviews', 'Company Jobs', 'Location', 'Company Description', 'Number of Employees', 'Industry Type'], axis = 1)
companydata3.head()

#Rename the columns.
companydata3 = companydata3.rename(columns={'Company Name': 'companyname', 'Company salaries': 'companysalary', 'Company rating':'companyrating'})

#Check for null/naan values.
companydata3.isna().sum()

#Remove naan/null.

companydata3.replace('-', np.nan, inplace = True)
companydata3 = companydata3.dropna()

#Recheck
companydata3.isna().sum()


# In[14]:


#Clean the company name
companydata3['companyname'][0].split('\n')[0]
# remove numerical features from company name
companydata3['companyname'] = companydata3['companyname'].apply(lambda x : x.split("\n")[0])

#Clean company salary
companydata3['companysalary'] = companydata3['companysalary'].str.split(",").str[0].str.replace('K','').str.split('-').str[0].str.replace('$','').str.replace('&', '').str.replace('(no)', '').str.replace('-', '').str.replace('^','')

#Review cleaned dataframe
companydata3.head()


# In[15]:


#remove rows that are duplicated
companydata3.drop_duplicates(keep=False, inplace=False)


# In[16]:


#Merge the two dataframes on company name! Remove non matching.

CCDF = pd.merge(jobsdf, companydata3, on ='companyname')
CCDF.shape


# The amount of data that we lost by merging by company name was quite large, however, the data was only comparable by company name. To include the other companies could cause an inaccurate analysis.

# In[17]:


#Check combined dataframe for duplicate rows
#remove rows that are duplicated
CCDF.drop_duplicates(keep=False, inplace=False)


# In[18]:


#This signifancly reduced our datas size but as we know GIGO! Garbage in produces garbage out. Cleaning is important.
CCDF.shape


# In[19]:


#Check for categorical and data types. In linear regression, we can not use categorical features until they are converted for testing purposes. 
CCDF.dtypes


# In[20]:


#Run get dummies on categorical variables
#One hot encoding with get dummies must be completed for modeling. String data can not be used. So we will use this for our categorical/string values to have integers as our result. This will allow multiple linear regression.

CCDF = pd.get_dummies(CCDF, columns = ['jobtitleClean'])
CCDF.head()


# In[21]:


#Rename get dummies, 
CCDF = CCDF.rename(columns={'jobtitleClean_analyst': 'Analyst', 'jobtitleClean_engineer': 'Engineer', 'jobtitleClean_scientist': 'Scientist', 'jobtitleClean_senior roles': 'Senior_Roles'})

#remove rows that are duplicated
CCDF.drop_duplicates(keep=False, inplace=False)

#Review finalized cleaned dataframe.
CCDF.head()


# In[22]:


#Run a profile report to see how the cleaned data looks and if I am good to save and proceed. 
# run the profile report
profile = CCDF.profile_report(title='Pandas Profiling Report')
profile

# save the report as html file
profile.to_file(output_file="pandas_profiling1.html")
   
# save the report as json file
profile.to_file(output_file="pandas_profiling2.json")


# In[23]:


#Save cleaned dataframe before splitting.
CCDF.to_csv('Documents/CCDF.CLEANED.BeforeSplit.csv')


# I will attach the summary profile above to my submission. This Panda's profile report is great for linear regression.It provides so many details and outlooks on the entire dataset. Even with cleaning and removing it identified possible duplicate rows. Many of the variables that still remain in my data are highly correlated. I will not be keeping all of the columns for the actual model and predictions. I have kept data now the exploratory data analysis. However the only columns needed for our research is the average salary, rating, size and company name.
# 

# # D - Data Analysis (with EDA)

# For exploratory data analysis, I used Seaborn and Matplotlib along with a few simple Panda's features to compare each variable to the average salary. I selected this techniques as they were the most straight forward and left little room for misunderstanding. 
# 
# An advantage of the techniques I used is that they are very simple and leave little room for misinterpreting the data. They are straight to the point. 
# 
# A disadvantage to this is that it is less visually pleasing than some more technical graphs and plots that could have been used. 
# 
# First, I am taking a look at the salary column, broken down into minimum, maximum, and average. Matplotlib allows me to use a simple plot to show a bar chart of the variable broken down. 
# 
# For modeling, I chose to utilize Multiple Linear Regression. I chose it because I felt like it was the best way to compare multiple explanatory variables against one dependent variable. 
# I chose to use Statsmodels OLS for my model. This method is the most simple and leaves little room for error on the user's end. It utilizes a formula API function and provides a number of statistical models. It sets up linear regression and provides summary information. A disadvantage to this is that it does not evaluate the results for you. 
# 
# After using Statsmodels OLS, I performed a gradient-boosting regression with GridSearchCV. This allowed me to test several models such as Lasso, ElasticNet, KNeighbors, DecisionTree, and SVR. I utilized a simple imputer to pull multiple model results. The advantage of this was the ability to compare multiple models at once. 

# In[24]:


jobsdf['min_sal'].hist()
plt.title('Min Salary')
plt.show()

jobsdf['max_sal'].hist()
plt.title('Max Salary')
plt.show()



jobsdf['avg_salary'].hist()
plt.title('Avg. Salary')
plt.show()


# In[25]:


job_title_salary = jobsdf['avg_salary'].groupby(jobsdf['jobtitleClean']).mean().round(0).nlargest(15).sort_values(ascending = False).reset_index()
plt.figure(figsize=(25,9))
fig, ax = plt.subplots()
ax = sns.barplot(ax = ax, data = job_title_salary , y = job_title_salary.jobtitleClean, x = job_title_salary.avg_salary)
ax.set(ylabel='Job titles',xlabel='Salary', title='Average Salaries by Job Titles')
ax.bar_label(ax.containers[0], padding = 2)

jobsdf[["jobtitleClean", "avg_salary"]]
jobsdf[["jobtitleClean", "avg_salary"]].groupby(["jobtitleClean"], as_index= False).mean().sort_values(by= "avg_salary", ascending= False)


# Above, when we break down the job titles and compare them to the average salary, We can see the average salary of a scientist is around 106k, senior roles are at 102k, engineers are 99k,and analysts are around 72k.
# 
# Since we are looking at what is more likely for companies to base their decisions on, we will only use the average salary. The above graph helps us identify that the average salaries listed are in the 50k–130k range.

# In[26]:


#For simple review purposes, checking out the min and max salaries.
jobsdf[["jobtitleClean", "max_sal"]]
jobsdf[["jobtitleClean", "max_sal"]].groupby(["jobtitleClean"], as_index= False).mean().sort_values(by= "max_sal", ascending= False)


# In[27]:


jobsdf[["jobtitleClean", "min_sal"]]
jobsdf[["jobtitleClean", "min_sal"]].groupby(["jobtitleClean"], as_index= False).mean().sort_values(by= "min_sal", ascending= False)


# In[28]:


#Reviewing the average salary by year
jobsdf[["jobtitleClean", "Year", "avg_salary"]]
jobsdf[["jobtitleClean", "Year", "avg_salary"]].groupby(["Year"], as_index= False).mean().sort_values(by= "avg_salary", ascending= False)


# In[29]:


pd.pivot_table(jobsdf,index = ['jobtitleClean','Year'], values = 'avg_salary')


# When seeing the data we pulled from Glassdoor, it appears analyst salaries in 2021 were almost $7,000 higher and then reduced. Engineers, scientists, and senior roles increased quite a bit. Engineers average salaries increased by 28k, scientists salaries increased by 34k, and senior roles increased by 39k. This shows us an average of a 31K increase in 3 years.
# 
# Now I will look at how large the company is in comparison to average salaries.

# In[30]:


jobsdf[["size", "avg_salary"]]
jobsdf[["size", "avg_salary"]].groupby(["size"], as_index= False).mean().sort_values(by= "avg_salary", ascending= False)


# I expected a much different result. Company size is broken down on a scale of 1–7, with 1 being the smallest and 7 for companies with over 10,000 employees. This scale shows a large variance in company size and average salary.

# Reviewing the companies with the lowest average salary and then the highest average salary.

# In[31]:


jobsdf.nsmallest(n=20, columns=['avg_salary'])


# In[32]:


jobsdf.nlargest(n=20, columns=['avg_salary'])


# Seeing the above shows that Fleetcor, NPD, Veterans Affairs, Text Health, Mcphail Associates, and Beebee Healthcare are significantly below all other companies. While Liberty Mutual, Gallup, Sage Intaact, CA-One Tech Cloud, Grand Ronds, The Climate Corporation, Visa, and Tapjoy are clearly on the higher end.
# 
# Below, we review the average salary by company rating.

# In[33]:


toprated3 = CCDF['rating'].groupby(jobsdf['avg_salary']).mean().round(0).nlargest(15).sort_values(ascending = False).reset_index()
plt.figure(figsize=(25,10))
fig, ax = plt.subplots()
ax = sns.barplot(ax = ax, data = toprated3 , y = toprated3.avg_salary, x = toprated3.rating)
ax.set(ylabel='rating',xlabel='avg_salary', title='Average Salary by Company Rating')
ax.bar_label(ax.containers[0], padding = 2)


# The result does show that the higher the company rating, the higher the salary. This is important to see because if employees are satisfied with their salary, they are more likely to be satisfied with their employer.
# 
# Now that we have reviewed the exploratory analysis and identified the differences, We will move on to visuals that represent our modeling.
# 
# First, I will run Seaborn regplots for all of our variables. I need to visualize my variables before plotting. I will need to check for linearity first. Linearity is when there is a visible relationship that exists between dependent and independent variables. 
# 
# I will review each variable against the average salary. 
# 
# For rating, it appears there are many outliers. It also appears to be very linear. The higher the rating, the higher the average salary, but it appears to stop around a four rating and not increase. I will still keep this in the model, but after visualizing, my expectations are not very high.
# 
# Size is quite similar to rating in linearity. There is an upward trend, but it does not appear to be impacted by size.
# 
# People with the job title of analyst appear to have a downward trend in salary. While people with an engineering title tend to have a slight curve upward.
# 
# Scientists and senior roles also show a minimal upward trend, but it is still upward.

# In[34]:


sns.regplot(x="rating", y="avg_salary", data=CCDF, line_kws={'color':'red'})
plt.show()
sns.regplot(x="size", y="avg_salary", data=CCDF, line_kws={'color':'red'})
plt.show()
sns.regplot(x="Analyst", y="avg_salary", data=CCDF, line_kws={'color':'red'})
plt.show()
sns.regplot(x="Engineer", y="avg_salary", data=CCDF, line_kws={'color':'red'})
plt.show()

sns.regplot(x="Scientist", y="avg_salary", data=CCDF, line_kws={'color':'red'})
plt.show()

sns.regplot(x="Senior_Roles", y="avg_salary", data=CCDF, line_kws={'color':'red'})
plt.show()


# In[35]:


sns.regplot(x="avg_salary", y="Analyst", data=CCDF, line_kws={'color':'red'})
plt.show()


# Spearman's correlation is used in statistics, as a nonparametric alternative to Pearson's correlation. Spearman's is best used with continuous variables to determine if they exhibit a monotonic relationship. If a value results in.00-.19, it is very weak;.20-.39 is weak;.40-.59 is moderate;.70-.79 is strong, and.80-1.0 is very strong. Results between -1 and 1 are considered perfect. A positive result means a relationship between two variables in which both move in the same direction. A negative correlation means there is a relationship where one variable decreases when the other increases. (4. Frost, 2023)

# In[36]:


#Check independence of each variable with Spearman Coefficient
rating= CCDF["rating"]
avg_salary = CCDF["avg_salary"]
#Find the Spearmen Cofficient.
spearmanr_coff, p_value = spearmanr(rating,avg_salary)
spearmanr_coff


# In[37]:


rating= CCDF["size"]
avg_salary = CCDF["avg_salary"]
#Find the Spearmen Cofficient.
spearmanr_coff, p_value = spearmanr(rating,avg_salary)
spearmanr_coff


# In[38]:


rating= CCDF["Analyst"]
avg_salary = CCDF["avg_salary"]
#Find the Spearmen Cofficient.
spearmanr_coff, p_value = spearmanr(rating,avg_salary)
spearmanr_coff


# In[39]:


rating= CCDF["Engineer"]
avg_salary = CCDF["avg_salary"]
#Find the Spearmen Cofficient.
spearmanr_coff, p_value = spearmanr(rating,avg_salary)
spearmanr_coff


# In[40]:


rating= CCDF["Scientist"]
avg_salary = CCDF["avg_salary"]
#Find the Spearmen Cofficient.
spearmanr_coff, p_value = spearmanr(rating,avg_salary)
spearmanr_coff


# In[41]:


rating= CCDF["Senior_Roles"]
avg_salary = CCDF["avg_salary"]
#Find the Spearmen Cofficient.
spearmanr_coff, p_value = spearmanr(rating,avg_salary)
spearmanr_coff


# All our results were betwee -1 and 1, which means a very strong correlation of our variables in relationship to average salary.

# # Multiple Linear Regression

# Before I start my modeling analysis, I want to remove the columns that are not necessary. I will only leave the variables we are testing and average salary.

# In[42]:


CCDF = CCDF.drop(['companyname', 'Year', 'min_sal', 'max_sal', 'companyrating', 'companysalary'], axis = 1)

CCDF.head()


# In[43]:


#Review the correlation matrix
# correlation matrix
corr = CCDF.corr()

#print cor
corr
plt.figure(figsize=(16,8))
sns.heatmap(corr,cmap='rainbow',annot=True)

plt.show()


# (C1. Seaborn, N.D.) A correlation matrix using a heat map is a great way to see the correlation between variables at a quick glance. A -1 indicates a perfectly negative linear correlation between the variables, 0 indicates no linear correlation, and 1 indicates a perfect positive. The further away from the 0 the coefficient is, the stronger the relationship between the two variables. This shows us that majority of our variables are quite correlated. Seaborn has a great plot feature that allows for easy production of the heatmap by using an Axes-level function that draws into the current in the most recent provided argument.

# Our data was quite large when I initally downloaded it from Kaggle. After combining and cleaning it has significantly reduced. 

# In[44]:


CCDF.shape


# In[45]:


# creating feature variables
X = CCDF.drop('avg_salary',axis= 1)
y = CCDF[['rating', 'size', 'Analyst', 'Engineer','Scientist', 'Senior_Roles']]
# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

# creating a regression model
model = LinearRegression()

# fitting the model
model.fit(X_train,y_train)

# making predictions
predictions = model.predict(X_test)

#model evaluation
print(
  'MSE : ', MSE(y_test, predictions))
print(
  'MAE : ', MAE(y_test, predictions))


# In[46]:


r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")


# In[47]:


y_pred = model.predict(X)
print(f"predicted response:\n{y_pred}")


# For my multiple linear regression model, I will be using backward-elimination. This is completed by selecting a significance level. I have chosen the standard .05. Next, I fit the the full model with all possible predictors (job title, rating, size). 
# After, I run the model I look at the predictor with the highest P-Value. If P is greater than .05 standard significance, I remove it and re-run the model without that variable.(5. Geeksforgeeks, 2023)
# 

# In[48]:


#Multiple Regression Model
# (First regression model. Split the data into two different data sets with a 7:3 ratio. Since some columns have smaller integer ratios, this is important. 
from sklearn.model_selection import train_test_split
# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
CCDF_train, CCDF_test = train_test_split(CCDF, train_size = 0.7, test_size = 0.3, random_state = 100)
#Next – rescale the features. This does not include the dummy variables.  Rescaling is very important as the features all need  comparable scale. If not, that could set the coefficients results will not be accurate.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Applying scaler() to all the columns except the 'dummy' variables
num_vars = ['rating', 'size', 'avg_salary']
CCDF_train[num_vars] = scaler.fit_transform(CCDF_train[num_vars])
#Divide the data into X and Y
y_train = CCDF_train.pop('avg_salary')
X_train = CCDF_train
#Build a linear model and add all variables
import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train)
lr_1 = sm.OLS(y_train, X_train_lm).fit()
lr_1.summary()


# The model suggest there may be strong multicollinearity between variables. From our prior review it did appear that rating and size were highly correlated. This also leads me to believe highest p-values that will appear first will be rating and size. When running this first model we can clearly see that rating has a .420 p-value. This will be the first removed in our backwards elimination.

# In[49]:


#create a new dataframe with only the variables kept

CCDF2 = CCDF.drop(['rating'], axis = 1)
CCDF2


# In[50]:


# (Second regression model. Split the data into two different data sets with a 7:3 ratio. Since some columns have smaller integer ratios, this is important. 
from sklearn.model_selection import train_test_split
# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
CCDF2_train, CCDF2_test = train_test_split(CCDF2, train_size = 0.7, test_size = 0.3, random_state = 100)

# creating feature variables
X2 = CCDF2.drop('avg_salary',axis= 1)
y2 = CCDF2[['size', 'Analyst', 'Engineer','Scientist', 'Senior_Roles']]

# creating train and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.3, random_state=101)

# creating a regression model
model = LinearRegression()

# fitting the model
model.fit(X_train2,y_train2)

# making predictions
predictions = model.predict(X_test2)

y_pred2 = model.predict(X2)
print(f"predicted response:\n{y_pred2}")


# In[51]:


#Next – rescale the features. This does not include the dummy variables.  Rescaling is very important as the features all need  comparable scale. If not, that could set the coefficients results will not be accurate.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['size', 'avg_salary']
CCDF2_train[num_vars] = scaler.fit_transform(CCDF2_train[num_vars])
CCDF2_train
#Divide the data into X and Y
y_train2 = CCDF2_train.pop('avg_salary')
X_train2 = CCDF2_train

#Build a linear model and add all variables
import statsmodels.api as sm
X_train2_lm = sm.add_constant(X_train2)
lr_1 = sm.OLS(y_train2, X_train2_lm).fit()
lr_1.summary()


# Now that we removed rating, it appears that size could be what is still causing the alert for strong multicollinearity. R-Squared, AIC, and BIC all improved after removing rating. So we will remove size as it is higher than the .05 standard signficance we are using.

# In[52]:


#create a new dataframe with only the variables kept

CCDF3 = CCDF2.drop(['size'], axis = 1)
CCDF3


# In[53]:


# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
CCDF3_train, CCDF3_test = train_test_split(CCDF3, train_size = 0.7, test_size = 0.3, random_state = 100)
# creating feature variables
X3 = CCDF3.drop('avg_salary',axis= 1)
y3 = CCDF3[['Analyst', 'Engineer','Scientist', 'Senior_Roles']]
print(X3)
print(y3)
# creating train and test sets
X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X3, y3, test_size=0.3, random_state=101)

#Next – rescale the features. This does not include the dummy variables.  Rescaling is very important as the features all need  comparable scale. If not, that could set the coefficients results will not be accurate.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# creating a regression model
model3 = LinearRegression()

# fitting the model
model3.fit(X_train3,y_train3)

# making predictions
predictions3 = model3.predict(X_test3)

y_pred3 = model3.predict(X3)
print(f"predicted response:\n{y_pred3}")


# In[54]:


# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['avg_salary']
CCDF3_train[num_vars] = scaler.fit_transform(CCDF3_train[num_vars])
CCDF3_train

#Divide the data into X and Y
y_train3 = CCDF3_train.pop('avg_salary')
X_train3 = CCDF3_train

#Build a linear model and add all variables
import statsmodels.api as sm
X_train3_lm = sm.add_constant(X_train3)
lr_1 = sm.OLS(y_train3, X_train3_lm).fit()
lr_1.summary()



# Since I removed all the P-Values and multicollinearity still exists, I will only use one job title to compare the data. Since each of the job title columns are still job titles, they become categories, and I only need one for an accurate comparison of the variables. (6.Zach, 2021) I will now remove the three other job titles and complete model comparison based off analyst alone for my linear regression model.

# In[55]:


#create a new dataframe with only the variables kept

CCDF4 = CCDF.drop(['Engineer', 'Scientist', 'Senior_Roles'], axis = 1)
CCDF4


# In[56]:


# Split the data into two different data sets with a 7:3 ratio. Since some columns have smaller integer ratios, this is important. 
from sklearn.model_selection import train_test_split
# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
CCDF4_train, CCDF4_test = train_test_split(CCDF4, train_size = 0.7, test_size = 0.3, random_state = 100)

# creating feature variables
X4 = CCDF4.drop('avg_salary', axis=1)
y4 = CCDF4[['rating', 'size', 'Analyst']]

# creating train and test sets
X_train4, X_test4, y_train4, y_test4 = train_test_split(
    X4, y4, test_size=0.3, random_state=101)

# creating a regression model
model = LinearRegression()

# fitting the model
model.fit(X_train4,y_train4)

# making predictions
predictions = model.predict(X_test4)

y_pred4 = model.predict(X4)
print(f"predicted response:\n{y_pred4}")


# In[57]:


# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['avg_salary']
CCDF4_train[num_vars] = scaler.fit_transform(CCDF4_train[num_vars])
CCDF4_train

#Divide the data into X and Y
y_train4 = CCDF4_train.pop('avg_salary')
X_train4 = CCDF4_train

#Build a linear model and add all variables
import statsmodels.api as sm
X_train4_lm = sm.add_constant(X_train4)
lr_1FOUR = sm.OLS(y_train4, X_train4_lm).fit()
lr_1FOUR.summary()


# It is clear that having job titles divided up was causing the multi-collinearity issue. I am only using the 'analyst' job title as our job title difference. This removed the multicollinearity issue. I now have variables rating and size with a high p-value. 

# In[58]:


#create a new dataframe with only the variables kept

CCDF5 = CCDF4.drop(['size'], axis = 1)
CCDF5


# In[59]:


# Split the data into two different data sets with a 7:3 ratio. Since some columns have smaller integer ratios, this is important. 
from sklearn.model_selection import train_test_split
# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
CCDF5_train, CCDF5_test = train_test_split(CCDF5, train_size = 0.7, test_size = 0.3, random_state = 100)

# creating feature variables
X5 = CCDF5.drop('avg_salary', axis=1)
y5 = CCDF5[['rating', 'Analyst']]

# creating train and test sets
X_train5, X_test5, y_train5, y_test4 = train_test_split(
    X5, y5, test_size=0.3, random_state=101)

# creating a regression model
model = LinearRegression()

# fitting the model
model.fit(X_train5,y_train5)

# making predictions
predictions = model.predict(X_test5)

y_pred5 = model.predict(X5)
print(f"predicted response:\n{y_pred5}")


# In[60]:


# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['avg_salary']
CCDF5_train[num_vars] = scaler.fit_transform(CCDF5_train[num_vars])
CCDF5_train

#Divide the data into X and Y
y_train5 = CCDF5_train.pop('avg_salary')
X_train5 = CCDF5_train

#Build a linear model and add all variables
import statsmodels.api as sm
X_train5_lm = sm.add_constant(X_train5)
lr_1FIVE = sm.OLS(y_train5, X_train5_lm).fit()
lr_1FIVE.summary()


# It appears we still need to remove rating from our model.

# In[61]:


#create a new dataframe with only the variables kept

CCDF6 = CCDF5.drop(['rating'], axis = 1)
CCDF6


# In[62]:


# Split the data into two different data sets with a 7:3 ratio. Since some columns have smaller integer ratios, this is important. 
from sklearn.model_selection import train_test_split
# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
CCDF6_train, CCDF6_test = train_test_split(CCDF6, train_size = 0.7, test_size = 0.3, random_state = 100)

# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['avg_salary']
CCDF6_train[num_vars] = scaler.fit_transform(CCDF6_train[num_vars])
CCDF6_train

# creating feature variables
X6 = CCDF6.drop('avg_salary', axis=1)
y6 = CCDF6[['Analyst']]

# creating train and test sets
X_train6, X_test6, y_train6, y_test6 = train_test_split(
    X6, y6, test_size=0.3, random_state=101)

# creating a regression model
model6 = LinearRegression()

# fitting the model
model.fit(X_train6,y_train6)


# In[63]:


# fitting the model
model.fit(X_test6,y_test6)

# making predictions
predictions = model.predict(X_test6)

y_pred6 = model.predict(X6)
print(f"predicted response:\n{y_pred6}")


# In[64]:


# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['avg_salary']
CCDF6_train[num_vars] = scaler.fit_transform(CCDF6_train[num_vars])
CCDF6_train

#Divide the data into X and Y
y_train6 = CCDF6_train.pop('avg_salary')
X_train6 = CCDF6_train

#Build a linear model and add all variables
import statsmodels.api as sm
X_train6_lm = sm.add_constant(X_train6)
lr_1SIX = sm.OLS(y_train6, X_train6_lm).fit()
lr_1SIX.summary()


# In[65]:


#initiate linear regression model
model = LinearRegression()

#define predictor and response variables
X, y = CCDF6[["avg_salary", "Analyst"]], CCDF6.avg_salary

#fit regression model
model.fit(X, y)

#calculate R-squared of regression model
r_squared = model.score(X, y)

#view R-squared value
print(r_squared)


# In[66]:


Yhat = model.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])


# In[67]:


#Plotting the mean square error of average salary and predicted value using multifit.
plt.plot(CCDF[['avg_salary', 'Analyst']])
plt.xlabel('avg_salary')
plt.ylabel('Analyst')
plt.show()


# In[68]:


model.intercept_
print('What is the value of the interept (a)?', model.intercept_)


# In[69]:


model.coef_
print('What is the value of slope b?', model.coef_)


# In[70]:


MSE = MSE(CCDF6["avg_salary"], Yhat)
print("The mean square of average salary and job title is: ", MSE)


# In[71]:


#Plotting the mean square error of average salary and predicted value using multifit.
plt.plot(CCDF6['avg_salary'], Yhat)
plt.xlabel('Actuals')
plt.ylabel('Predicted')
plt.show()


# In[72]:


CCDF6.corr()


# In[73]:


#Time to try other tests for comparison.
X = CCDF[['avg_salary', 'Analyst', 'Scientist', 'Engineer', 'Senior_Roles']]
Y = X
X = X.drop(('avg_salary'), axis = 1).values


# In[74]:


train,val=train_test_split(CCDF)
X,y=train.drop('avg_salary',axis=1),train.avg_salary
train.shape,val.shape,X.shape,y.shape


# In[75]:


lm = LinearRegression()
lm.fit(X_train, y_train)
# neg_mean_absolute_error shows how much I am off from general prediction on the average.

np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3))


# In[76]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3))


# In[77]:


# Now I will try other models (C2. Holdsworth, 2021)
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet())) 
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor())) 
models.append(('SVR', SVR()))
models.append(('rf', RandomForestRegressor()))


# In[78]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
results1 = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=0, shuffle=True)
    pipe=Pipeline([('imputer',imputer),('model',model)])
    cv_results = cross_val_score(pipe, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    results1.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[79]:


df1=pd.DataFrame(data=[result.mean() for result in results1],index=names,columns=['Baseline'])
df1


# In[80]:


from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(n_quantiles=341)
imputer=SimpleImputer()

results2 = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=0, shuffle=True)
    pipe=Pipeline([('imputer',imputer),('qt',qt),('model',model)])
    cv_results = cross_val_score(pipe, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    results2.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[81]:


df2=pd.DataFrame(data=[result.mean() for result in results2],index=names,columns=['QuantileTransformer'])
df2


# In[82]:


df3=df1.join(df2)
df3


# In[83]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

imputer=SimpleImputer()
skb = SelectKBest(score_func=f_classif, k=6)

results3 = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=0, shuffle=True)
    pipe=Pipeline([('imputer',imputer),('skb',skb),('model',model)])
    cv_results = cross_val_score(pipe, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    results3.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print  (msg)


# In[84]:


df4=pd.DataFrame(data=[result.mean() for result in results3],index=names,columns=['SelectKBest'])
df4


# In[85]:


df5=df3.join(df4)
df5


# In[86]:


X=SimpleImputer().fit_transform(X)

ensembles = []
ensembles.append(('AB', AdaBoostRegressor()))
ensembles.append(('GBM', GradientBoostingRegressor()))
ensembles.append(('RF', RandomForestRegressor(n_estimators=10)))
ensembles.append(('ET', ExtraTreesRegressor(n_estimators=10)))

results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=10, random_state=0, shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[87]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor()
parameters = {'learning_rate': [0.01,0.02,0.03,0.04],
                  'subsample'    : [0.9, 0.5, 0.2, 0.1],
                  'n_estimators' : [100,500,1000, 1500],
                  'max_depth'    : [4,6,8,10]}
grid_GBR = GridSearchCV(estimator=GBR, param_grid = parameters, cv = 2, n_jobs=-1)
grid_GBR.fit(X, y)


# In[88]:


print("Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_GBR.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_GBR.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_GBR.best_params_)


# # Data Summary and Implications

# ### E: Summarize the implications of my data analysis/results
# Job titles significantly and statistically impact data science salaries. Company size and company rating are not statistically significant in data science salaries.
# 
# Exploring EDA could lead you to believe that company size and rating are impactful. Running Spearman's correlation on every variable proved that each variable had a significant correlation. Plotting company rating side by side with salary, it did show a higher salary with a higher rating. However, rating and size may have been highly correlated with each other, causing multicollinearity issues. Scatterplots of size and rating also did not show a significant increase or decrease in salary. Based on the model, size and rating are not significant to salary.
# 
# The results of my OLS linear regression model were 25% R-Squared.  26% of the average salary can be determined by job title. The Log-Likelihood is 383. The higher this number, the better the model. It's obtained by finding the parameter that maximizes the log-likelihood of the observed sample. AIC is determined by K, the number of model parameters, and In (L), the log likelihood of the model. The model with the lowest AIC offers the best fit. The value is not important. The BIC tried to find the model with the most truth among the candidates. Both of these are higher than our other models, validating that this is the most accurate model using Multiple Linear Regression with OLS.
# 
# My model used backwards elimination to reduce variables by a P-value higher than my 0.5 significance level. Rating and size were removed. 
# 
# A limitation of my analysis was that I still experiencing a very high degree of multicollinearity. This was identified due to the job title being split up into multiple categories. To eliminate the multicollinearity issue, I chose to use only the Analyst column, as this still explains the relationship between job title, size, rating, and salary.
# 
# Some of the data we analyzed showed that there was a strong increase in salaries from 2021 to 2023. Only the job title of analyst decreased. All other roles increased by 7,000. This would lead me to recommend that employers re-analyze analyst salaries for 2024. 
# 
# The average salary of a scientist is around 106k, senior roles are at 102k, engineers are 99k,and analysts are around 72k. I also recommend employers use this as an average pay for these roles. It appears that the rating of the company and size are not factors in salary variance, and I would not recommend employers base salary on these variables. It appears job titles have a significant impact on salaries, and by reviewing the trends in this research, employers can accurately position themselves in a competitive market. 
# 
# I propose for future analysis that a different method, such as Principal Component analysis, be used to analyze salaries versus job title, rating, and size. I would also recommend that future research include a location variable to see if that has an impact as well.
# 
# 
# 

# ###  Sources
# 1. Adam Raelson. (2018, May 9). Why and how you should be using Glassdoor - Glassdoor for employers. Why and how you should be using Glassdoor. https://www.glassdoor.com/employers/blog/why-and-how-you-should-be-using-glassdoor/ 
# 2. Apollo Technical Engineered Talent Solutions. (2023, April 9). 19 employee retention statistics that will surprise you (2023 ). Apollo Technical LLC. https://www.apollotechnical.com/employee-retention-statistics/ 
# 3. Tutorials Point. (n.d.). Employee retention - challenges. Online Courses and eBooks Library. https://www.tutorialspoint.com/employee_retention/employee_retention_challenges.htm#:~:text=Challenges%20in%20Employee%20Retention&amp;text=Salary%20Dissatisfaction%20%E2%88%92%20Every%20employee%20has,the%20budget%20of%20the%20organization. 
# 4. Frost, J. (2023, July 1). Spearman’s correlation explained. Statistics By Jim. https://statisticsbyjim.com/basics/spearmans-correlation/ 
# 5. GeeksforGeeks. (2023, January 25). ML: Multiple linear regression using python. GeeksforGeeks. https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/ 
# 6. Zach. (2021, October 21). A guide to multicollinearity &amp; VIF in regression. Statology. https://www.statology.org/multicollinearity-regression/#:~:text=If%20you%20determine%20that%20you%20do%20need%20to,analysis%20or%20partial%20least%20squares%20%28PLS%29%20regression.%20 
# 
# ### Coding Sources
# C1. Seaborn. (n.d.). Seaborn.heatmap#. seaborn.heatmap - seaborn 0.12.2 documentation. https://seaborn.pydata.org/generated/seaborn.heatmap.html 
# C2. Holdsworth, P. (2021, July 4). Gradientboostingregressor + GRIDSEARCHCV. Kaggle. https://www.kaggle.com/code/paulh2718/gradientboostingregressor-gridsearchcv 

# In[ ]:




