import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import datetime
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.dates import date2num
import gc
plt.style.use('ggplot')
#suppress annoying warnings about pandas assignment
pd.options.mode.chained_assignment = None
#suppress all warnings
import warnings
warnings.filterwarnings("ignore")
#format x axis with dates
def format_x_date(ax):
    #set ticks every week
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


def get_country_data(selection):
    selection = selection.iloc[:,4:].T
    selection.columns=['Death']
    selection['Death_change']=selection['Death']-selection['Death'].shift(1)
    selection=selection.reset_index()
    selection['index'] = pd.to_datetime(selection['index'])
    selection['Death_change_log']=np.log10(selection['Death_change'])
    selection['Death_change_log']=selection['Death_change_log'].replace(-np.inf,np.nan)
    selection=selection[~selection['Death_change_log'].isna()]
    selection.columns=['date']+list(selection.columns[1:])
    selection['ndate']=date2num(selection['date'])
    return selection

def make_fig1(df):
    plt.style.use('ggplot')
    before = df[df['data']<'2020-03-9']
    before['ndata']=date2num(before['data'])
    after = df[(df['data']>'2020-03-9') & (df['data']<'2020-03-24') ]
    after['ndata']=date2num(after['data'])

    after2 = df[(df['data']>='2020-03-24')  ]
    after2['ndata']=date2num(after2['data'])

    x='ndata'
    y='deceduti_daily_log'
    fig,ax=plt.subplots(figsize=(8,8))
    sns.regplot(x=x, y=y, color="r", data=before,ax=ax,label='before')
    sns.regplot(x=x, y=y, color="b", data=after,ax=ax,label='15 days after ')
    sns.regplot(x=x, y=y, color="black", data=after2,ax=ax,label='25 days after')
    format_x_date(ax)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin-1,xmax+1)
    #plt.xticks(rotation=70,ha='right')
    plt.xlabel('Date', fontsize='16')
    plt.ylabel('Deaths (log10)',fontsize='16')
    plt.title('Death Trend',fontsize='18')
    plt.legend(title='Social Distancing',fontsize=12)
    plt.savefig('Fig1.png')
    plt.show()
    
    
#plt.style.use('ggplot')

def despine(ax,log_y=False):
    small_add=1.1   
    ymin, ymax = ax.get_ylim()
    if log_y:
        small_add = np.log10(small_add) 
    #print(ymin, ymax)
    #print(small_add)
    #ax.set_ylim(ymin+small_add,ymax)

    ax.spines['left'].set_bounds(ymin+small_add, ymax)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    xmin, xmax = ax.get_xlim()
    #ax.set_xlim(xmin,xmax)

    ax.spines['bottom'].set_bounds(xmin+0.5, xmax)
    
def lollipop_plot(df, ax, date_col='data', value_col='not_admitted_ratio',
                 color='#007acc',label='Ratio'):
    #make lollipop
    ax.plot(df[date_col].dt.date, df[value_col], "o",
        markersize=5, color=color, alpha=0.9,label=label)
    #make stick
    ax.vlines(x=df[date_col].dt.date, ymin=0, 
          ymax=df[value_col], color=color, alpha=0.2, linewidth=5)

    #set ticks every week
    despine(ax)
    format_x_date(ax)
    
    return ax

def my_plot(df, ax, date_col='data', 
            #value_3='ricoverati_con_sintomi_change',
            value_2='terapia_intensiva_change',
            value_1='deceduti_daily',
            
            label_1='Deaths',
            label_2="Intensive Care",
            #label_3="Hospital Admissions",
           ):
    #make lollipop
    #ax.plot(df[value_3], df[date_col].dt.date, "o",
    #    markersize=5, color='green', alpha=0.9,label=label_3)    
    ax.plot(df[value_2], df[date_col].dt.date, "o",
        markersize=5, color='blue', alpha=0.9,label=label_2)
    ax.plot(df[value_1], df[date_col].dt.date, "o",
        markersize=5, color='red', alpha=0.9,label=label_1)

    
    xmin=df[[value_1,value_2]].min(axis=1)
    xmax=df[[value_1,value_2]].max(axis=1)
    #make stick
    ax.hlines(y=df[date_col].dt.date, xmin=xmin,
              xmax=xmax, color='black', alpha=0.2, linewidth=2)

    
    #ymin, ymax = ax.get_ylim()
    #ax.set_ylim(ymin,ymax)
    #ax.spines['left'].set_bounds(ymin, ymax-0.5)
    despine(ax)
    #set ticks every 3 days
    #ax.yaxis.set_major_locator(mdates.WeekdayLocator())
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=3))
    #set major ticks format
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    #ax.spines['top'].set_color('none')
    #ax.spines['right'].set_color('none')
    #xmin, xmax = ax.get_xlim()
    #ax.spines['bottom'].set_bounds(0, xmax-1)
    return ax

def make_fig2(df):
    plt.style.use('default')
    from matplotlib.patches import Ellipse

    fig,axes=plt.subplots(figsize=(16,6
                                  ),ncols=3)

    ax=axes[1]
    ax = my_plot(df, ax)
    df['ndate']=date2num(df['data'])

    ax.set_xlabel('Cases', fontsize='16')
    ax.set_ylabel('Date', fontsize='16')

    ax.annotate('Social Distancing Measures', 
                (df['deceduti_daily'][14]+20, mdates.date2num(df['data'][13])+0.2 ),
                xytext=(50, 2), 
                textcoords='offset points', 
                arrowprops=dict(arrowstyle='-|>'))
    ax.text(-0.1, 1.1, 'B)', horizontalalignment='center', 
                   verticalalignment='center',
                   transform=ax.transAxes,fontsize=16)

    ax.legend(loc=2, bbox_to_anchor=(0.5, 0.3), title='Infected')
    
    ax=axes[0]
    ax = lollipop_plot(df, ax, date_col='data',
                       value_col='terapia_intensiva',label='Intensive Care')

    ax.legend(loc=2, bbox_to_anchor=(0.1, 0.8), title='Infected')

    ax.set_xlabel('Date', fontsize='16')
    ax.set_ylabel('Admissions', fontsize='16')
    ax.text(-0.1, 1.1, 'A)', horizontalalignment='center', 
                   verticalalignment='center',
                   transform=ax.transAxes,fontsize=16)


    ax=axes[2]
    before = df[df['data']<'2020-03-9']
    before['ndata']=date2num(before['data'])
    after = df[(df['data']>'2020-03-9') & (df['data']<'2020-03-24') ]
    after['ndata']=date2num(after['data'])

    after2 = df[(df['data']>='2020-03-24')  ]
    after2['ndata']=date2num(after2['data'])

    x='ndata'
    y='deceduti_daily_log'
    #fig,ax=plt.subplots(figsize=(8,8))
    sns.regplot(x=x, y=y, color="r", data=before,ax=ax,label='before')
    sns.regplot(x=x, y=y, color="b", data=after,ax=ax,label='15 days after ')
    sns.regplot(x=x, y=y, color="black", data=after2,ax=ax,label='25 days after')


    despine(ax,log_y=True)
    format_x_date(ax)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin-1,xmax+1)


    ax.set_xlabel('Date', fontsize='16')
    ax.set_ylabel('Deaths (log10)',fontsize='16')
    ax.text(-0.1, 1.1, 'C)', horizontalalignment='center', 
                   verticalalignment='center',
                   transform=ax.transAxes,fontsize=16)

    plt.tight_layout()
    ax.legend(loc=2, bbox_to_anchor=(0.5, 0.3), title='Lockdown Days')
    plt.suptitle('Death vs Intensive Care Admissions',fontsize='18',y=1.1)
    plt.savefig('Fig2.png')
    plt.show()

def make_fig3(df,allDf):
    plt.style.use('default')
    #sns.palplot(sns.color_palette("Accent"))
    palette=sns.color_palette("Accent")
    days_after_lock=15

    fig,ax=plt.subplots(figsize=(8,8))
    temp = df[df['data']>'2020-03-9']
    temp['days_from_lockdown']=np.arange(0,df[df['data']>'2020-03-9'].shape[0])

    sns.regplot(x='days_from_lockdown',y='deceduti_daily_log'
                ,data=temp.iloc[0:days_after_lock],fit_reg=True,color=palette[0],
               marker='.',line_kws={'linestyle':'--'},ci=None) 
    sns.regplot(x='days_from_lockdown',y='deceduti_daily_log',
                data=temp.iloc[days_after_lock:],fit_reg=True,color=palette[0],
                label='Italy',marker='.',ci=None) 


    spain_df = allDf[(allDf['Country/Region']=='Spain')]
    spain_df =  get_country_data(spain_df)
    spain_df = spain_df[spain_df['date']>='2020-03-16']
    spain_df['days_from_lockdown']=np.arange(0,spain_df.shape[0])


    sns.regplot(x='days_from_lockdown',y='Death_change_log'
                ,data=spain_df.iloc[0:days_after_lock],fit_reg=True,color=palette[1],
                marker='.',line_kws={'linestyle':'--'},ci=None) 
    sns.regplot(x='days_from_lockdown',y='Death_change_log',
                data=spain_df.iloc[days_after_lock:],fit_reg=True,color=palette[1],
                label='Spain',marker='.',ci=None) 


    uk_df = allDf[(allDf['Country/Region']=='United Kingdom') & (allDf['Lat']==55.3781)  ]
    uk_df =  get_country_data(uk_df) 
    uk_df = uk_df[uk_df['date']>='2020-03-23']
    uk_df['days_from_lockdown']=np.arange(0,uk_df.shape[0])


    sns.regplot(x='days_from_lockdown',y='Death_change_log'
                ,data=uk_df.iloc[0:days_after_lock],fit_reg=True,color=palette[-1],marker='.',
               line_kws={'linestyle':'--'},ci=None) 
    sns.regplot(x='days_from_lockdown',y='Death_change_log',
                data=uk_df.iloc[days_after_lock:],fit_reg=True,
                color=palette[-1],label='UK',marker='.',ci=None) 





    ax.set_xlim(-2,temp.shape[0]+2)
    ax.set_ylabel('Deaths (log10)',fontsize='16')
    ax.set_xlabel('Days from Lockdown',fontsize='16')
    ax.legend(title='Country ',fontsize=12,loc=2)
    
    despine(ax,log_y=True)
    plt.title('Country Comparison',fontsize='18')
    plt.tight_layout()
    plt.savefig('Fig3.png')
#format_x_date(ax)