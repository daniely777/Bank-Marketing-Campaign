#!/usr/bin/env python
# coding: utf-8

# ## Module installation

# In[1]:


# pip install matplotlib


# In[2]:


# pip install scipy


# In[3]:


# pip install statsmodels


# In[4]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from calendar import month_name
from scipy.stats import chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind


# # 

# ## Import and handling of the dataset

# In[5]:


#First hands-on check with the dataset

df = pd.read_csv('dataset.csv')
df.info()


# In[6]:


#Assessing the amount of valiables per object column, and standarizing the data

for column_name, column_type in df.dtypes.items():
    if column_type == 'O':
        df[column_name] = df[column_name].apply(str.lower)
        print(len(df[column_name].unique()), df[column_name].unique())


# In[7]:


print(df.head())


# In[8]:


#Creating a new column to make the call_duration more friendly

seconds_to_minutes = [math.ceil(item/60) for item in df.call_duration]
df.insert(8, 'seconds_to_minutes', seconds_to_minutes)

print(df.head())


# In[9]:


#Assessing the amount and rates of converted clients

previous_converted = len(df[df.previous_campaign_outcome == 'successful'])
current_converted = len(df[df.conversion_status == 'converted'])
concurrent_converted = len(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'converted')])
current_converted_rate = round(current_converted/len(df)*100, 2)
conversion_increase_rate = round((current_converted/previous_converted*100)-100, 2)
concurrent_converted_rate = round(concurrent_converted/previous_converted*100, 2)

print(f'{previous_converted} clients were converted last campaign.')
print(f'{current_converted} clients were converted on current campaign. That\'s a {conversion_increase_rate}% increase in conversion rate from previous results and a {current_converted_rate}% of the dataset.')
print(f'{concurrent_converted} of the clients that were converted on last campaign, are staying converted on current one. That\'s a retain of {concurrent_converted_rate}% of the previous converted clients.')


# In[10]:


#Standarization of colors to be used
cnc,nc,nr,c,cc = ['#FF0000','#FF9933','#FFFF66','#B2FF66','#00FF00']


# In[11]:


#Barchar of previous campaign coversions and current one
#Barchar of previous converted and concurrent ones

plt.figure(figsize=(2*4, 4))
plt.subplot(1,2,1)
plt.title('Conversion results')
plt.bar(['Previous Converted', 'Current Converted'], [previous_converted, current_converted], width=0.5, edgecolor='black', color=['lightblue',c])
plt.subplot(1,2,2)
plt.title('Concurrently converted clients')
plt.bar(['Previous Converted', 'Concurrent Converted'], [previous_converted, concurrent_converted], width=0.5, edgecolor='black', color=['lightblue',cc])
plt.tight_layout()
plt.show()


# ## The plan

# In[12]:


#Setting up a plan

print(f'Seeing this campaign has been a success with a x3.5 increase in conversions, it\'s a good start to keep the clients we have converted.\nEven more so, seeing the percentage of retained customer has been {concurrent_converted_rate}% so far, at {concurrent_converted} clients.')
print(f'Keeping this retaining rate in mind, there is a potential of {round(current_converted*concurrent_converted_rate/100)} clients being further retained from this campagin, or at least {round((current_converted-previous_converted)*concurrent_converted_rate/100)} clients being retained from this campaign if the previous ones did not renew the next one.')


# In[13]:


#Plotting results from the dataset and checking for visual cues

color_converted = [c if item == 'converted' else nc for item in df.conversion_status]
amount_to_display = 20000

plt.figure(figsize=(4*3.5, 3*3.5))
for i in range(len(df.columns)):
    plt.subplot(3,4,i+1)
    plt.title(df.columns[i])
    plt.scatter(df.conversion_status[:amount_to_display], df[df.columns[i]][:amount_to_display], color=color_converted[:amount_to_display], alpha=0.002)

plt.tight_layout()  
plt.show()


# In[14]:


#Assesing first visuals cues

print('There appear to be certain trends and ranges for the age, call duration and call frequency for both converted and non-converted clients')


# # 

# ## The converted clientele, visual representation

# In[15]:


#Histograms of the amounts of each occupation, age, marital status, call duration, call frequency, that are currently converted and concurrently converted, against non-converted and not renewed
##we can then assess where to aim the efforts of the next campaign, as well as developing new strategies for the non-converted and not renewed clientele

plt.figure(figsize=(2.5*6, 4*8))
plt.suptitle('Distribution of converted clients', y=0.91, fontsize=20)
plt.subplot(5,1,1)
plt.title('Occupation')
plt.hist(df[df['conversion_status']=='converted']['occupation'], len(df.occupation.unique()), label='Converted', ec='black', orientation='horizontal', color=c)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'converted')]['occupation'], len(df.occupation.unique()), label='Concurrent converted', ec='black', orientation='horizontal', color=cc)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'not_converted')]['occupation'], len(df.occupation.unique()), label='Not renewed', ec='black', orientation='horizontal', color=nr)
plt.xlabel('Amount')
plt.yticks(np.arange(0.5, 11, 0.915))
plt.xticks(range(0,1400, 50))
plt.legend()


plt.subplot(5,1,2)
plt.title('Age')
plt.hist(df[df['conversion_status']=='converted']['age'], len(df.age.unique()), label='Converted', orientation='horizontal',  rwidth=0.9, ec='black', lw=0.2, color=c)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'converted')]['age'], len(df.age.unique()), label='Concurrent converted', orientation='horizontal', rwidth=0.9, ec='black', lw=0.2, color=cc)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'not_converted')]['age'], len(df.age.unique()), label='Not renewed', orientation='horizontal',  rwidth=0.9, ec='black', lw=0.2, color=nr)
plt.ylabel('Age')
plt.xlabel('Amount')
plt.xticks(range(0,240, 10))
plt.xlim(0,230)
plt.legend()

plt.subplot(5,1,3)
plt.title('Marital Status')
plt.hist(df[df['conversion_status']=='converted']['marital_status'], len(df.marital_status.unique()), label='Converted', orientation='horizontal',  rwidth=0.7, ec='black', color=c)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'converted')]['marital_status'], len(df.marital_status.unique()), label='Concurrent converted', orientation='horizontal', rwidth=0.7, ec='black', color=cc)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'not_converted')]['marital_status'], len(df.marital_status.unique()), label='Not renewed', rwidth=0.7, ec='black', orientation='horizontal', color=nr)
plt.xlabel('Amount')
plt.yticks(np.arange(0.33, 2, 0.66))
plt.xticks(range(0,2900, 150))
plt.legend()

plt.subplot(5,1,4)
plt.title('Call duration in minutes')
plt.hist(df[df['conversion_status']=='converted']['seconds_to_minutes'], len(df.seconds_to_minutes.unique()), label='Converted', orientation='horizontal', rwidth=0.9, ec='black', lw=0.3, color=c)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'converted')]['seconds_to_minutes'], len(df.seconds_to_minutes.unique()), label='Concurrent converted', orientation='horizontal', height=0.9, rwidth=0.9, ec='black', lw=0.3, color=cc)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'not_converted')]['seconds_to_minutes'], len(df.seconds_to_minutes.unique()), label='Not renewed', orientation='horizontal', height=0.6, rwidth=0.9, ec='black', lw=0.3, color=nr)
plt.ylabel('Minutes')
plt.xlabel('Amount')
plt.yticks(range(0,35, 5))
plt.xticks(range(0,650, 25))
plt.xlim(0, 600)
plt.ylim(-1,35)
plt.legend()

plt.subplot(5,1,5)
plt.title('Call frequency')
plt.hist(df[df['conversion_status']=='converted']['call_frequency'], len(df.call_frequency.unique()), label='Converted', ec='black', lw=0.3, orientation='horizontal', color=c)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'converted')]['call_frequency'], len(df.call_frequency.unique()), label='Concurrent converted', height=0.9, ec='black', lw=0.3, orientation='horizontal', color=cc)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'not_converted')]['call_frequency'], len(df.call_frequency.unique()), label='Not renewed', ec='black', orientation='horizontal', height=0.6, lw=0.3, color=nr)
plt.ylabel('Times')
plt.xlabel('Amount')
plt.yticks(range(0,21, 1))
plt.xticks(range(0,2650, 100))
plt.ylim(0,20)
plt.legend()

plt.show()


# ## The potential clientele, visual representation

# In[16]:


#Histograms of the amounts of each occupation, age, marital status, call duration, call frequency, of non-converted, concurrently non-converted, and not renewed clients
##we can then assess where to aim the efforts of the next campaign, as well as developing new strategies for not renewed clientele and people adamant to not be converted

plt.figure(figsize=(2.5*6, 4*8))
plt.suptitle('Distribution of non-converted clients', y=0.91, fontsize=20)
plt.subplot(5,1,1)
plt.title('Occupation')
plt.hist(df[df['conversion_status']!='converted']['occupation'], len(df.occupation.unique()), label='Non-converted', ec='black', orientation='horizontal', color=nc)
plt.hist(df[(df.previous_campaign_outcome == 'unsuccessful') & (df.conversion_status == 'not_converted')]['occupation'], len(df.occupation.unique()), label='Concurrent non-converted', ec='black', orientation='horizontal', color=cnc)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'not_converted')]['occupation'], len(df.occupation.unique()), label='Not renewed', ec='black', orientation='horizontal', color=nr)
plt.xlabel('Amount')
plt.yticks(np.arange(0.5, 11, 0.915))
plt.xticks(range(0,9001, 500))
plt.legend()

plt.subplot(5,1,2)
plt.title('Age')
plt.hist(df[df['conversion_status']!='converted']['age'], len(df.age.unique()), label='Non-converted', orientation='horizontal',  rwidth=0.9, ec='black', lw=0.2, color=nc)
plt.hist(df[(df.previous_campaign_outcome == 'unsuccessful') & (df.conversion_status == 'not_converted')]['age'], len(df.age.unique()), label='Concurrent non-converted', orientation='horizontal',  rwidth=0.9, ec='black', lw=0.2, color=cnc)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'not_converted')]['age'], len(df.age.unique()), label='Not renewed', orientation='horizontal',  rwidth=0.9, ec='black', lw=0.2, color=nr)
plt.ylabel('Age')
plt.xlabel('Amount')
plt.xticks(range(0,1901, 100))
plt.legend()

plt.subplot(5,1,3)
plt.title('Marital Status')
plt.hist(df[df['conversion_status']!='converted']['marital_status'], len(df.marital_status.unique()), label='Non-converted', rwidth=0.7, ec='black', orientation='horizontal', color=nc)
plt.hist(df[(df.previous_campaign_outcome == 'unsuccessful') & (df.conversion_status == 'not_converted')]['marital_status'], len(df.marital_status.unique()), label='Concurrent non-converted', rwidth=0.7, ec='black', orientation='horizontal', color=cnc)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'not_converted')]['marital_status'], len(df.marital_status.unique()), label='Not renewed', rwidth=0.7, ec='black', orientation='horizontal', color=nr)
plt.xlabel('Amount')
plt.yticks(np.arange(0.33, 2, 0.66))
plt.xticks(range(0,25001, 1500))
plt.legend()

plt.subplot(5,1,4)
plt.title('Call duration in minutes')
plt.hist(df[df['conversion_status']!='converted']['seconds_to_minutes'], len(df.seconds_to_minutes.unique()), label='Non-converted', orientation='horizontal', rwidth=0.9, ec='black', lw=0.3, color=nc)
plt.hist(df[(df.previous_campaign_outcome == 'unsuccessful') & (df.conversion_status == 'not_converted')]['seconds_to_minutes'], len(df.seconds_to_minutes.unique()), label='Concurrent non-converted', orientation='horizontal', rwidth=0.9, ec='black', lw=0.3, color=cnc)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'not_converted')]['seconds_to_minutes'], len(df.seconds_to_minutes.unique()), label='Not renewed', orientation='horizontal', height=0.6, rwidth=0.9, ec='black', lw=0.3, color=nr)
plt.ylabel('Minutes')
plt.xlabel('Amount')
plt.xticks(range(0,14001, 1000))
plt.ylim(-1,30)
plt.legend()

plt.subplot(5,1,5)
plt.title('Call frequency')
plt.hist(df[df['conversion_status']!='converted']['call_frequency'], len(df.call_frequency.unique()), label='Non-converted', ec='black', orientation='horizontal', lw=0.3, color=nc)
plt.hist(df[(df.previous_campaign_outcome == 'unsuccessful') & (df.conversion_status == 'not_converted')]['call_frequency'], len(df.call_frequency.unique()), label='Concurrent non-converted', ec='black', orientation='horizontal', height=0.6, lw=0.3, color=cnc)
plt.hist(df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'not_converted')]['call_frequency'], len(df.call_frequency.unique()), label='Not renewed', ec='black', orientation='horizontal', height=0.6, lw=0.3, color=nr)
plt.ylabel('Times')
plt.xlabel('Amount')
plt.xticks(range(0,27000, 2000))
plt.ylim(0,30)
plt.legend()

plt.show()


# # 

# ## Addressing the monthly results

# In[17]:


#Let's look into the outcome of each month

month_lookup = list(item.lower() for item in month_name)
# print(month_lookup[1:])
df_month = df.copy()
df_month.call_month = sorted(df_month.call_month, key=month_lookup[1:].index)

plt.figure(figsize=(20, 6))
plt.hist(df_month[df_month['conversion_status'] != 'converted']['call_month'], len(df_month.call_month.unique()), orientation='horizontal', ec='black', label='Non-converted', lw=0.3, color=nc)
plt.hist(df_month[df_month['conversion_status'] == 'converted']['call_month'], len(df_month.call_month.unique()), orientation='horizontal', ec='black', label='Converted', lw=0.3, color=c)
plt.hist(df_month[(df_month.previous_campaign_outcome == 'successful') & (df_month.conversion_status == 'converted')]['call_month'], len(df.call_month.unique()), orientation='horizontal', ec='black', label='Concurrent Converted', lw=0.3, color=cc)
plt.hist(df_month[(df_month.previous_campaign_outcome == 'unsuccessful') & (df_month.conversion_status != 'converted')]['call_month'], len(df.call_month.unique()), orientation='horizontal', ec='black', label='Concurrent non-converted', lw=0.3, color=cnc)
plt.hist(df_month[(df_month.previous_campaign_outcome == 'successful') & (df_month.conversion_status != 'converted')]['call_month'], len(df.call_month.unique()), orientation='horizontal', ec='black', label='Not renewed', lw=0.3, color=nr)
plt.title('Call month outcomes')
plt.xscale('linear')
plt.yticks(np.arange(0.5, 11, 0.915))
# plt.xticks(range(0,12001, 1000))
plt.xlabel('Amount')
plt.legend()
plt.show()


# In[18]:


#Determining the top months

c_top_call_month = df[df['conversion_status'] == 'converted']['call_month'].value_counts().reset_index(name='counts').sort_values('counts',ascending=False)
cc_top_call_month = df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'converted')]['call_month'].value_counts().reset_index(name='counts').sort_values('counts',ascending=False)
nc_top_call_month = df[df['conversion_status'] != 'converted']['call_month'].value_counts().reset_index(name='counts').sort_values('counts',ascending=False)
cnc_top_call_month = df[(df.previous_campaign_outcome == 'unsuccessful') & (df.conversion_status != 'converted')]['call_month'].value_counts().reset_index(name='counts').sort_values('counts',ascending=False)
nr_top_call_month = df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status != 'converted')]['call_month'].value_counts().reset_index(name='counts').sort_values('counts',ascending=False)

print(f'The top 3 months for successful calls are: {str(c_top_call_month.to_numpy()[[0,1,2],0]).title()[1:-1]}\nAnd the top 3 for concurrent clients are: {str(cc_top_call_month.to_numpy()[[0,1,2],0]).title()[1:-1]}')
print(f'The top 3 months for unsuccessful calls are: {str(nc_top_call_month.to_numpy()[[0,1,2],0]).title()[1:-1]}\nAnd the top 3 for concurrent non-converted clients are: {str(cnc_top_call_month.to_numpy()[[0,1,2],0]).title()[1:-1]}')
print(f'The top 3 months for not renewed clients are: {str(nr_top_call_month.to_numpy()[[0,1,2],0]).title()[1:-1]}')


# In[19]:


print(f'It is worth looking into the success rate throughout each month:')

#Creating a new dataframe to represent the success rates

nc_vs_c_call_month = nc_top_call_month.sort_values(['call_month'])
nc_vs_c_call_month['converted_calls'] = c_top_call_month.sort_values(['call_month'])['counts']
nc_vs_c_call_month.rename(columns={'counts':'non_converted_calls'}, inplace=True)
nc_vs_c_call_month['success_%'] = nc_vs_c_call_month.apply(lambda x: round(x.converted_calls/(x.non_converted_calls + x.converted_calls)*100, 2), axis=1)
nc_vs_c_call_month_sorted = nc_vs_c_call_month.copy()
nc_vs_c_call_month_sorted.call_month = sorted(nc_vs_c_call_month.call_month, key=month_lookup.index)
nc_vs_c_call_month_sorted = nc_vs_c_call_month_sorted.reset_index(drop=True)

print(nc_vs_c_call_month_sorted)


# In[20]:


#Visual representation of the success rate for each month, ordered by success rate (desc)

plt.figure(figsize=(4*3.5, 3*3.5))
plt.suptitle('Month and conversion success rate', y=1 , fontsize=20)
for i in range(len(df.call_month.unique())):
    plt.subplot(3,4,i+1)
    plt.title(f"{str(nc_vs_c_call_month.sort_values(['success_%'], ascending=False).to_numpy()[[i],0]).title()[2:-2]}: {str(nc_vs_c_call_month.sort_values(['success_%'], ascending=False).to_numpy()[[i],3])[1:-1]}%")
    plt.pie([int(str(nc_vs_c_call_month.sort_values(["success_%"], ascending=False).to_numpy()[[i],1])[1:-1]), int(str(nc_vs_c_call_month.sort_values(["success_%"], ascending=False).to_numpy()[[i],2])[1:-1])], labels=[f'Non-converted:\n{str(nc_vs_c_call_month.sort_values(["success_%"], ascending=False).to_numpy()[[i],1])[1:-1]}', f'Converted:\n{str(nc_vs_c_call_month.sort_values(["success_%"], ascending=False).to_numpy()[[i],2])[1:-1]}'], colors=[nc,c], shadow=1, startangle=45)
    plt.annotate(f'Total calls: {int(str(nc_vs_c_call_month.sort_values(["success_%"], ascending=False).to_numpy()[[i],1])[1:-1]) + int(str(nc_vs_c_call_month.sort_values(["success_%"], ascending=False).to_numpy()[[i],2])[1:-1])}', (0,0), (-1.5,-1.5) , annotation_clip=True)

plt.tight_layout()  
plt.show()

print(f'\nAs we can see, despite the peak months of conversion, the success rate was inversely proportional to the amount of calls made.\nI.e.: There is missing potential and the contact efforts should be re-distributed for each month towards the next campaign.')


# # 

# 
# ## Hyphotesis Testing

# In[21]:


#Displaying general stats, sorted by conversion status

print('General non-converted stats:\n', df[df['conversion_status'] != 'converted'].describe())
print('\nGeneral converted stats:\n', df[df['conversion_status'] == 'converted'].describe())
print('\nGeneral concurrent converted stats:\n', df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'converted')].describe())
print('\nGeneral not renewed stats:\n', df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status != 'converted')].describe())
print('\nGeneral concurrent non-converted stats:\n', df[(df.previous_campaign_outcome == 'unsuccessful') & (df.conversion_status != 'converted')].describe())


# ### Hypothesis test "Ages and conversion"

# In[22]:


#Hypothesis test "Ages and conversion": Is there are relation between the Age and the converted/concurrent converted/non-converted/not renewed status?
## Null hyp pval >= 0.05: We cannot say there is no relation, the same distribution of ages can represent these groups of clientele.
## Alt hyp pval < 0.05: We can say there is no relation. Ages vary from these groups of clientele.

nc_age = df[df['conversion_status'] != 'converted']['age']
c_age = df[df['conversion_status'] == 'converted']['age']
cc_age = df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status == 'converted')]['age']
nr_age = df[(df.previous_campaign_outcome == 'successful') & (df.conversion_status != 'converted')]['age']
cnc_age = df[(df.previous_campaign_outcome == 'unsuccessful') & (df.conversion_status != 'converted')]['age']
nc_age_mean = np.mean(nc_age)
nc_age_std = np.std(nc_age)
c_age_mean = np.mean(c_age)
c_age_std = np.std(c_age)
cc_age_mean = np.mean(cc_age)
cc_age_std = np.std(cc_age)
nr_age_mean = np.mean(nr_age)
nr_age_std = np.std(nr_age)
cnc_age_mean = np.mean(cnc_age)
cnc_age_std = np.std(cnc_age)

print('Let\'s check the different types of conversion status\' means and stds.\n')
print(f'{nc_age_mean}, {nc_age_std}, mean and std of non-converted.')
print(f'{c_age_mean}, {c_age_std}, mean and std of converted.')
print(f'{cc_age_mean}, {cc_age_std}, mean and std of concurrent converted.')
print(f'{nr_age_mean}, {nr_age_std}, mean and std of not renewed.')
print(f'{cnc_age_mean}, {cnc_age_std}, mean and std of concurrent non-converted.')

print('\nLet\'s now check if the std between the groups is valid enough for a test. From what we can see, the only ones worth checking would be converted, concurrent converted and not renewed.\nA value between 0.9 and 1.1 would be optimal:')
print(f'{c_age_std/cc_age_std}, between converted and concurrent converted.')
print(f'{c_age_std/nr_age_std}, between converted and not renewed.')
print(f'{cc_age_std/nr_age_std}, between concurrent converted and not renewed.')
print(f'{nc_age_std/cnc_age_std}, between concurrent converted and not renewed.')


# In[23]:


print('Seeing that 4 out of the 5 groups could be somewhat related, we will perform a Tukey\'s Range Test to further confirm this.\n')

conversion_ages = np.concatenate((nc_age, c_age, cc_age, nr_age, cnc_age))
conversion_labels = ['nc']*len(nc_age) + ['c']*len(c_age) + ['cc']*len(cc_age) + ['nr']*len(nr_age) + ['cnc']*len(cnc_age)

turkey_results = pairwise_tukeyhsd(conversion_ages,conversion_labels,0.05)
print(turkey_results)


# In[24]:


#Hypothesis test 'Ages and conversion' results:

print('We can conclude that, out of the 5 groups tested:\n')
print('- We cannot reject the Null hypothesis for the Converted and Not Renewed groups, their age distribution does not vary enough to say they are not related.\n')
print('- We cannot reject the Null hypothesis for all 3 Non-converted, Concurrently Non-converted and Not Renewed groups, their age distribution does not vary enough to say they are not related.\n')
print('\nThe rest of the groups\'s age distributions vary enough to reject the Null hypothesis, indicating there was no detected relation between their group\'s ages.\n')


# ### Hypothesis test "Education level and conversion"

# In[25]:


#Hypothesis test "Education level and conversion": Is there association between the level of studies and the likelyhood of conversion?
## Null hyp pval >= 0.05: We cannot say there is no association.
## Alt hyp pval < 0.05: We can say there is no association.
#'high_school' 'unidentified' 'college' 'elementary_school'

nc_hs = len(df[(df.education_level == 'high_school') & (df.conversion_status != 'converted')])
c_hs = len(df[(df.education_level == 'high_school') & (df.conversion_status == 'converted')])
nc_u = len(df[(df.education_level == 'unidentified') & (df.conversion_status != 'converted')])
c_u = len(df[(df.education_level == 'unidentified') & (df.conversion_status == 'converted')])
nc_col = len(df[(df.education_level == 'college') & (df.conversion_status != 'converted')])
c_col = len(df[(df.education_level == 'college') & (df.conversion_status == 'converted')])
nc_el = len(df[(df.education_level == 'elementary_school') & (df.conversion_status != 'converted')])
c_el = len(df[(df.education_level == 'elementary_school') & (df.conversion_status == 'converted')])

contingency_ed = [[nc_hs, c_hs], [nc_u, c_u], [nc_col, c_col], [nc_el, c_el]]
print(contingency_ed)

chi2, pval, dof, expected = chi2_contingency(contingency_ed)
print(pval)


# In[26]:


#Hypothesis test "Education level and conversion" results:

print('We can conclude that, out of the 4 groups tested:\n')
print('- We can confidently reject the Null hypothesis. There is no association between the education level and conversion of the clientele.')


# ### Hypothesis test "Marital status and conversion"

# In[27]:


#Hypothesis test "Marital status and conversion": Is there association between the marital status and the likelyhood of conversion?
## Null hyp pval >= 0.05: We cannot say there is no association.
## Alt hyp pval < 0.05: We can say there is no association.

nc_m = len(df[(df.marital_status == 'married') & (df.conversion_status != 'converted')])
c_m = len(df[(df.marital_status == 'married') & (df.conversion_status == 'converted')])
nc_div = len(df[(df.marital_status == 'divorced') & (df.conversion_status != 'converted')])
c_div = len(df[(df.marital_status == 'divorced') & (df.conversion_status == 'converted')])
nc_single = len(df[(df.marital_status == 'single') & (df.conversion_status != 'converted')])
c_single = len(df[(df.marital_status == 'single') & (df.conversion_status == 'converted')])

contingency_marital = [[nc_m, c_m], [nc_div, c_div], [nc_single, c_single]]
print(contingency_marital)

chi2, pval, dof, expected = chi2_contingency(contingency_marital)
print(pval)


# In[28]:


#Hypothesis test "Marital status and conversion" results:

print('We can conclude that, out of the 3 groups tested:\n')
print('- We can confidently reject the Null hypothesis. There is no association between the marital status and conversion of the clientele.')


# # 

# ## A/B Sample size calculation

# In[29]:


#Determining the minimum desired lift

print(f'Current campaigns\' conversion rate is {current_converted_rate}%\nSay we are to pilot-test the next campaign and want at least a 50% increase in conversion rate ({round(11.7*1.5,2)}%) for the next campaign, with a significance threshold of 5%:\nHow big would the sample have to be to test this out?')


# In[30]:


#Calculating the sample size

baseline_conversion = 11.7
minium_lift = 50
significance_threshold = 5
confidence = 95

print(f'We would require a sample size of 1,307 people to pilot-test the next campaign.')


# # 

# # CONCLUSIONS / REPORT

# In[31]:


#Gathering thoughts and proactive measures, based on the statistics

nc_seconds_to_minutes_sorted = np.sort(df[df['conversion_status'] != 'converted']['seconds_to_minutes'])
nc_seconds_to_minutes_80 = np.percentile(df[df['conversion_status'] != 'converted']['seconds_to_minutes'], 80)
nc_call_frequency_sorted = np.sort(df[df['conversion_status'] != 'converted']['call_frequency'])
nc_call_frequency_80 = np.percentile(df[df['conversion_status'] != 'converted']['call_frequency'], 80)
c_seconds_to_minutes_sorted = np.sort(df[df['conversion_status'] == 'converted']['seconds_to_minutes'])
c_seconds_to_minutes_80 = np.percentile(df[df['conversion_status'] == 'converted']['seconds_to_minutes'], 80)
c_call_frequency_sorted = np.sort(df[df['conversion_status'] == 'converted']['call_frequency'])
c_call_frequency_80 = np.percentile(df[df['conversion_status'] == 'converted']['call_frequency'], 80)


print('Here are the conclusions from this Bank marketing capaign analysis:\n\n')
print(f'A total of {previous_converted} clients were converted last campaign.')
print(f'A total of {current_converted} clients were converted on current campaign. That\'s a {conversion_increase_rate}% increase in conversion rate from previous results and a {current_converted_rate}% of the dataset.')
print(f'{concurrent_converted} of the clients that were converted on last campaign, are staying converted on current one. That\'s a retain of {concurrent_converted_rate}% of the previous converted clients.\n')

print(f'Seeing this campaign has been a success with a x2.5 increase in conversions, it\'s a good start to keep the clients we have converted.\nEven more so, seeing the percentage of retained customer has been {concurrent_converted_rate}% so far, at {concurrent_converted} clients.')
print(f'Keeping this retaining rate in mind, there is a potential of {round(current_converted*concurrent_converted_rate/100)} clients being further retained from this campagin, or at least {round((current_converted-previous_converted)*concurrent_converted_rate/100)} clients being retained from this campaign if the previous ones did not renew the next one.\n')

print('We can see that, besides the Occupation, the Age of the clients is fairly distributed with a noticeable drop for people 64 years old and older, and the Call Duration and Call Frequency are certainly related both in trend and depending on the converted/non-converted outcome.\n')
print('It is also worth mentioning that single and married clients tend to remain converted, as opposed to divorced ones.\n')
print('The occupation of the clients indicates a strong inclination for conversion for:\nExecutives\nTechnical Specialists\nAdministrative Staff\nManual Workers\nRetired Workers\n')
print('Business Owners, Manual Workers and Domestic Workers seem to be specially reluctant to stay sucessfully converted from one campaign to the next. Perhaps there could be more tailored programmes for them, given the uncertanties that can envelop such occupations.')
print('In constrast, we can see high rotation in concurrency of conversion on Administrative Staff and Technical Specialists positions (nearly as many of them stay converted than not renewing). This could mean they are more demanding, in terms of requiring more enticing campaigns to keep their fidelity.\n')
print('Nonetheless, there seems to be a correlation between the expected income from high job possitions, and the likelyhood to be enrolled in the Bank\'s marketing campaings. It is a strong point to pursue in further campaigns.\n')

print(f'A line could be set for the duration and frequency of the calls, taking into account the successful conversions\' data:\nThe 80% percentile for call duration and frequency of converted clients is {int(c_seconds_to_minutes_80)} minutes, and {int(c_call_frequency_80)} calls respectively.\nThis insight could be used as guide for the calls made to potential clients.\n')
print(f'The 80% percentile of call duration for unsucessfull calls is no longer than {int(nc_seconds_to_minutes_80)} minutes and no more than {int(nc_call_frequency_80)} calls.\nWe can see outliers in the call duration that have gone to even {nc_seconds_to_minutes_sorted[-2]} and {nc_seconds_to_minutes_sorted[-1]} minutes, and call frequency going to even {nc_call_frequency_sorted[-2]} and {nc_call_frequency_sorted[-1]} times.')
print(f'While the consistency in call duration and amount of calls is solid with an average of {round(np.mean(c_seconds_to_minutes_sorted))} minutes for successful calls and {round(np.mean(nc_seconds_to_minutes_sorted))} for the unsuccessful ones, having 20% of the calls stranding from the general lenght and amount means that human and time resources could be put to better use setting a hard limit for the duration and frequency of the campaign calls.\n')

print(f'As for the success of conversion calls for each month, despite the peak months of conversion ({str(c_top_call_month.to_numpy()[[0,1,2],0]).title()[1:-1]}), the success rate was inversely proportional to the amount of calls made.\nI.e.: There is missing potential and the contact efforts should be re-distributed for each month towards the next campaign.\n')

print('It has been concluded we cannot say there is no relation between Non-converted, Concurrently Non-converted and Not Renewed age groups, suggesting there could be trends for those group\'s ages that converge with this campaign\'s negative outcome, that should be looked into moving towards the next campaign.')
print('It has also been concluded that there is no statistical association between neither the education level and conversion of the clientele, nor their marital status.\n')
print('If we were to pilot-test the next campaign and wanted at least a 50% increase in conversion rate (17.55%), we would require a sample size of 1,307 people.\n')

print('\nThat concludes my analysis. All feedback is greatly appreciated, even more so as this is my first portafolio project.\nThank you for reading.\nAll the best,\n Daniel Y.')


# Bibliography
# 
# Dataset: https://www.kaggle.com/datasets/yaminh/bank-marketing-campaign-dataset
# 
# Calendar module: https://docs.python.org/3/library/calendar.html
# 
# A/B test sample size calculator: https://www.statsig.com/calculator?mde=50&bcr=11.7&twoSided=false&splitRatio=0.5&alpha=0.05&power=0.95
