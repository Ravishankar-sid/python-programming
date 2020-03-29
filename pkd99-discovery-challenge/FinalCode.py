# Importing required libraries
import pandas as pd
import datetime as dt
import plotly
import plotly.graph_objs as go

# Reading in Financial Data

AccDat = pd.read_csv('C:/Users/Jatan Dewgun/Downloads/MBD2018_FP_GroupAssignment_FinancialData/data_berka/account.asc', sep = ";")
CrdDat = pd.read_csv('C:/Users/Jatan Dewgun/Downloads/MBD2018_FP_GroupAssignment_FinancialData/data_berka/card.asc', sep = ";")
ClnDat = pd.read_csv('C:/Users/Jatan Dewgun/Downloads/MBD2018_FP_GroupAssignment_FinancialData/data_berka/client.asc', sep = ";")
DspDat = pd.read_csv('C:/Users/Jatan Dewgun/Downloads/MBD2018_FP_GroupAssignment_FinancialData/data_berka/disp.asc', sep = ";")
DstDat = pd.read_csv('C:/Users/Jatan Dewgun/Downloads/MBD2018_FP_GroupAssignment_FinancialData/data_berka/district.asc', sep = ";")
LoanDat = pd.read_csv('C:/Users/Jatan Dewgun/Downloads/MBD2018_FP_GroupAssignment_FinancialData/data_berka/loan.asc', sep = ";")
OrdDat = pd.read_csv('C:/Users/Jatan Dewgun/Downloads/MBD2018_FP_GroupAssignment_FinancialData/data_berka/order.asc', sep = ";")
TrnDat = pd.read_csv('C:/Users/Jatan Dewgun/Downloads/MBD2018_FP_GroupAssignment_FinancialData/data_berka/trans.asc', sep = ";", low_memory = False)

# Having a look at the data

AccDat.head()
CrdDat.head()
ClnDat.head()
DspDat.head()
DstDat.head()
LoanDat.head()
OrdDat.head()
TrnDat.head()

# Checking the Data Types for the Datasets

AccDat.dtypes
CrdDat.dtypes
ClnDat.dtypes
DspDat.dtypes
DstDat.dtypes
LoanDat.dtypes
OrdDat.dtypes
TrnDat.dtypes

# Building Functions to deploy during Data Preparation

# Function to calculate number of days from the start date of reference.
StartDate = dt.datetime(1993, 1, 1)


def NumDaysSD(x):
    n = x - StartDate
    return n.days

# Function to calculate number of days from the end date of reference


EndDate = dt.datetime(2000, 1, 1)


def NumDaysED(x):
    n = EndDate - x
    return n.days

# Function to calculate

# Function to convert 'Frequency' column values on Account Data to English


def language_conversion(x):
    if x == 'POPLATEK MESICNE':
        return 'MONTHLY'
    elif x == 'POPLATEK TYDNE':
        return 'WEEKLY'
    elif x == 'POPLATEK PO OBRATU':
        return 'TRANSACTION'
    else:
        return 'UNKNOWN'

# Some functions which will be incorporated to clean the 'date' variable on Clients Database


def ExtractMid(x):
    return int(x/100) % 100


def ExtractMonth(x):
    Month = ExtractMid(x)
    if Month > 50:
        return Month - 50
    else:
        return Month


def ExtractDay(x):
    return x % 100


def ExtractYear(x):
    return int(x/10000) + 1900


def ExtractGender(x):
    Month = ExtractMid(x)
    if Month > 50:
        return 'F'
    else:
        return 'M'


def DateConverter(x):
    Day = ExtractDay(x)
    Month = ExtractMonth(x)
    Year = ExtractYear(x)
    return dt.datetime(Year, Month, Day)


def BirthDateCalculator(x):
    Day = ExtractDay(x)
    Month = ExtractMonth(x)
    Year = ExtractYear(x)
    return NumDaysED(dt.datetime(Year, Month, Day))/365

# Function to convert the 'KSymbol' values on the Order Database to English


def OrdKSymConverter(x):
    if x == 'POJISTNE':
        return 'INSURANCE PAYMENT'
    elif x == 'SIPO':
        return 'HOUSEHOLD PAYMENT'
    elif x == 'LEASING':
        return 'LEASING PAYMENT'
    elif x == 'UVER':
        return 'LOAN PAYMENT'
    else:
        return 'UNKNOWN'

# Function to convert certain variables' values on the Transaction Database.
# We convert the values for 'Type', 'Operation' and 'KSymbol' variables.


def TrnTypeConverter(x):
    if x == 'PRIJEM':
        return 'CREDIT'
    elif x == 'VYDAJ':
        return 'WITHDRAWAL'
    else:
        return 'UNKNOWN'


def TrnOprConverter(x):
    if x == 'VYBER KARTOU':
        return 'CC WITHDRAWAL'
    elif x == 'VKLAD':
        return 'CREDIT IN CASH'
    elif x == 'PREVOD Z UCTU':
        return 'COLLECTION OTHER BANK'
    elif x == 'VYBER':
        return 'CASH WITHDRAWAL'
    elif x == 'PREVOD NA UCET':
        return 'REMITTANCE TO OTHER BANK'
    else:
        return 'UNKNOWN'


def TrnKSymConverter(x):
    if x == 'POJISTNE':
        return 'INSURANCE PAYMENT'
    elif x == 'SLUBZY':
        return 'STATEMENT PAYMENT'
    elif x == 'UROK':
        return 'CREDITED INTEREST'
    elif x == 'SANKC. UROK':
        return 'INTEREST SANCTION'
    elif x == 'SIPO':
        return 'HOUSEHOLD'
    elif x == 'DUCHOD':
        return 'PENSION'
    elif x == 'UVER':
        return 'PAYMENT LOAN'
    else:
        return 'UNKNOWN'

# We observe on the District database that there are certain observations specifically under
# the UnempRate95 and Crimes95 variables which are '?'. We take care of this using a function.


def QMarkConverter(x, type):
    if x == '?':
        return -1
    elif type == 'float':
        return float(x)
    else:
        return int(x)

# Building Recency, Frequency and Monetary Variables based on the Transaction Database


def TrnVariableBuilder(x):
    VBDate = dt.datetime(1999, 1, 1)
    x['TransactionRecency'] = (VBDate - x['TrnDate'].max()).days
    x['LOR'] = (x['TrnDate'].max() - x['TrnDate'].min()).days
    x['TransactionFrequency'] = x['TransactionID'].count()

    Credit = x.loc[(x.TrnType == 'CREDIT')]
    Withdrawal = x.loc[(x.TrnType == 'WITHDRAWAL')]

    x['MValTrnCred'] = Credit['TrnAmount'].mean()
    x['MValTrnWDR'] = Withdrawal['TrnAmount'].mean()
    return x


def OrdVariableBuilder(x):
    x['OrderFrequency'] = x['OrderID'].count()

    Lease = x.loc[(x.OrdKSymbol == 'LEASING PAYMENT')]
    Housing = x.loc[(x.OrdKSymbol == 'HOUSEHOLD PAYMENT')]
    Insurance = x.loc[(x.OrdKSymbol == 'INSURANCE PAYMENT')]

    x['FreqOrdLease'] = Lease['OrderID'].count()
    x['FreqOrdHousing'] = Housing['OrderID'].count()
    x['FreqOrdInsurance'] = Insurance['OrderID'].count()

    x['MValLease'] = Lease['OrderID'].mean()
    x['MValHousing'] = Housing['OrderID'].mean()
    x['MValInsurance'] = Insurance['OrderID'].mean()
    return x


# Cleaning the Data


# Renaming Columns
AccDat = AccDat.rename(columns = {'account_id':'AccountID','frequency':'StatementFreq', 'district_id' : 'AccDistID', 'date': 'AccOpenedOn'})

ClnDat = ClnDat.rename(columns = {'district_id':'DistID', 'client_id': 'ClientID'})

CrdDat = CrdDat.rename(columns = {'card_id':'CardID','disp_id':'DisponentID','type':'CardType', 'issued':'CrdIssueDate'})

DspDat = DspDat.rename(columns = {'disp_id':'DisponentID','client_id':'ClientID', 'account_id':'AccountID', 'type':'DspType'})

DstDat = DstDat.rename(columns={'A1': 'DistID','A2': 'DistName', 'A3':'Region','A4': 'Inhabitants', 
               'A5':'Municipalities<499', 'A6':'Muncipalities<500_1999', 'A7':'Municipalities<2000_9999',
               'A8':'Municipalities<10000','A9': 'NumCities','A10': 'RatioUrban','A11': 'AvgSalary', 
               'A12':'UnempRate95', 'A13':'UnempRate96', 'A14':'Entrep1000Inhab', 'A15':'Crimes95', 'A16':'Crimes96'})

LoanDat = LoanDat.rename(columns = {'loan_id': 'LoanID', 'account_id':'AccountID','date':'LoanDate', 'duration':'LoanDuration', 
                                    'payments':'MonthlyLoanPymt', 'status':'LoanStatus', 'amount': 'LoanAmount' })

OrdDat = OrdDat.rename(columns = {'order_id':'OrderID', 'account_id':'AccountID', 'bank_to': 'OrdBankTo', 
                                  'account_to':'OrdAccountTo', 'amount':'OrdAmount', 'k_symbol':'OrdKSymbol'})

TrnDat = TrnDat.rename(columns = {'trans_id':'TransactionID', 'account_id':'AccountID', 'date':'TrnDate', 
                                  'type':'TrnType', 'operation':'TrnOperation', 'amount':'TrnAmount', 'balance':'TrnBalance',
                                  'k_symbol':'TrnKSymbol', 'bank':'TrnBank', 'account':'TrnAccount'})

# Applying the created functions to clean our data.

AccDat['StatementFreq'] = AccDat['StatementFreq'].apply(language_conversion)
AccDat['AccOpenedOn'] = AccDat['AccOpenedOn'].apply(DateConverter)
AccDat['DaysAccOpenedOn'] = AccDat['AccOpenedOn'].apply(NumDaysSD)


ClnDat['ClientAge'] = ClnDat['birth_number'].apply(BirthDateCalculator)
ClnDat['Gender'] = ClnDat['birth_number'].apply(ExtractGender)
del ClnDat['birth_number']

CrdDat['CrdIssueDate'] = pd.to_datetime(CrdDat['CrdIssueDate'].str[:6], format = '%y%m%d')

DstDat['Crimes95'] = DstDat['Crimes95'].apply(QMarkConverter, args=('int',))
DstDat['UnempRate95'] = DstDat['UnempRate95'].apply(QMarkConverter, args=('float',))

LoanDat['LoanDate'] = LoanDat['LoanDate'].apply(DateConverter)
LoanDat['DaysLoan'] = LoanDat['LoanDate'].apply(NumDaysSD)

OrdDat['OrdKSymbol'] = OrdDat['OrdKSymbol'].apply(OrdKSymConverter)


TrnDat['TrnDate'] = TrnDat['TrnDate'].apply(DateConverter)
TrnDat['TrnTransactionDays'] = TrnDat['TrnDate'].apply(NumDaysSD)
TrnDat['TrnType'] = TrnDat['TrnType'].apply(TrnTypeConverter)
TrnDat['TrnOperation'] = TrnDat['TrnOperation'].apply(TrnOprConverter)
TrnDat['TrnKSymbol'] = TrnDat['TrnKSymbol'].apply(TrnKSymConverter)

# Aggregating the Data
# Here we aggregate the Transaction Data and the Order Data

TrnDat = TrnDat.groupby('AccountID').apply(TrnVariableBuilder)

TrnAgg = TrnDat.drop(['TransactionID', 'TrnDate', 'TrnOperation', 'TrnType', 'TrnTransactionDays',
                     'TrnAmount','TrnBalance','TrnKSymbol','TrnBank','TrnAccount'], axis=1).drop_duplicates()

OrdDat = OrdDat.groupby('AccountID').apply(OrdVariableBuilder)
OrdAgg = OrdDat.drop(['OrderID', 'OrdBankTo', 'OrdAccountTo', 'OrdAmount', 'OrdKSymbol'], axis = 1).drop_duplicates()


#End of Data Cleaning



# Merging the Preliminary Datasets to create the Final DataMart

Datamart = pd.merge(ClnDat, DspDat, on = 'ClientID', how = 'left')

Datamart = Datamart.merge(CrdDat, on = 'DisponentID', how = 'left')

Datamart = Datamart.merge(DstDat, on = 'DistID', how = 'left')

Datamart = Datamart.merge(AccDat, on = 'AccountID', how = 'left')

Datamart = Datamart.merge(OrdAgg, on = 'AccountID', how = 'left' )

Datamart = Datamart.merge(LoanDat, on = 'AccountID', how = 'left')

Datamart = Datamart.merge(TrnAgg ,on = 'AccountID', how = 'left')

Datamart.head()

# Distinguishing between Favourable and Non-Favourable Customers
# We Distinguish Customers based on their Loan Status, If the customers pay their loans and have no pending payments
# we can say that they are favourable customers otherwise we classify them as non-favourable customers 
Datamart.loc[Datamart['LoanStatus'].isin(['A','C']),'CustomerType'] = 1   
Datamart.loc[Datamart['LoanStatus'].isin(['B','D']),'CustomerType'] = 0
Datamart['CustomerType'] = Datamart['CustomerType'].map({1:'Favourable', 0:'Non-Favourable'})


Datamart.to_csv('C:/Users/Jatan Dewgun/Downloads/MBD2018_FP_GroupAssignment_FinancialData/data_berka/basetable.csv')

#End of Merging Data


#Plots

# Demographic Plots


#Clients Per Gender
MFBar = go.Bar(
    x=['Male','Female'],
    y=[sum(Datamart['Gender']=='M'), sum(Datamart['Gender']=='F')],
    text=[format(sum(Datamart['Gender']=='M')/len(Datamart['Gender']),"10.2%"),
          format(sum(Datamart['Gender']=='F')/len(Datamart['Gender']),"10.2%")],
    textposition = 'auto'
    )
Plotter = [MFBar]
LOMFBar = go.Layout(
    title='Clients Per Gender',
    xaxis=dict(
        title='Gender',
        titlefont=dict(
            family='Calibri, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Client Count',
        titlefont=dict(
            family='Calibri, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
PlotMF = go.Figure(data= Plotter, layout=LOMFBar)
plotly.offline.plot(PlotMF, filename='Clients Per Gender.html')


# Clients Per Age Group
Datamart['AgeGroup'] = Datamart['ClientAge'] // 10 * 10
client_pv = Datamart.pivot_table(index='AgeGroup', columns='Gender', values='ClientID', aggfunc='count')
AGFem = go.Bar(
    x=client_pv.index,
    y=client_pv['F'],
    name='Female'
)
AGMale = go.Bar(
    x=client_pv.index,
    y=client_pv['M'],
    name='Male'
)
AG = [AGFem, AGMale]
layout = go.Layout(
    barmode='group',
    title='Clients per Age Group per Gender'
)
PlotAG = go.Figure(data=AG, layout=layout)
plotly.offline.plot(PlotAG, filename='AgeGroupGender.html')

# Disponents and Owners
Datamart['AgeGroup'] = Datamart['ClientAge'] // 10 * 10
client_pv = Datamart.pivot_table(index='AgeGroup', columns='DspType', values='ClientID', aggfunc='count')
Owners = go.Bar(
    x=client_pv.index,
    y=client_pv['OWNER'],
    name='Owner'
)
Disponents = go.Bar(
    x=client_pv.index,
    y=client_pv['DISPONENT'],
    name='Disponent'
)
OD = [Owners, Disponents]
layout = go.Layout(
    barmode='group',
    title='Owners and Disponents per Age Group '
)
PlotOD = go.Figure(data=OD, layout=layout)
plotly.offline.plot(PlotOD, filename='OwnersDisponents.html')

# Regions
Regional = Datamart['Region']
Region = go.Histogram(
    x = Regional,
    opacity=0.75
)

RegTrace = [Region]
layout = go.Layout(barmode='overlay', title='Clients per Region')
PlotReg = go.Figure(data=RegTrace, layout=layout)
plotly.offline.plot(PlotReg, filename='Regional.html')

# Districts
DistrictWise = Datamart['DistName']
District = go.Histogram(
    x = DistrictWise,
    opacity = 0.75
)
DistTrace = [District]
layout = go.Layout(barmode='overlay', title='Clients per District')
PlotDist = go.Figure(data=DistTrace, layout=layout)
plotly.offline.plot(PlotDist, filename='DistrictWise.html')

# Customers Based on Card Type
CType = Datamart['CardType']
Cards = go.Histogram(
    x = CType,
    opacity = 0.75
)
CardTrace = [Cards]
layout = go.Layout(barmode='overlay', title='Clients per Card Type')
PlotCard = go.Figure(data=CardTrace, layout=layout)
plotly.offline.plot(PlotCard, filename='CType.html')




# Frequency and Monetary Value Graphs for different variables



#Order Frequency
FreqOrd = Datamart['OrderFrequency']
OrdFreq = go.Histogram(
    x = FreqOrd,
    opacity=0.75
)
OFTrace = [OrdFreq]
layout = go.Layout(barmode='overlay', 
                   title='Orders Made by Clients', 
                   bargroupgap = 0.1,
                   xaxis = go.layout.XAxis(
                                     ticktext=['Single Order', 'Two Orders', 'Three Orders', 'Four Orders', 'Five Orders'],
                                     tickvals= ['1' , '2', '3', '4', '5'])
    )
PlotOF = go.Figure(data=OFTrace, layout=layout)
plotly.offline.plot(PlotOF, filename='FreqOrd.html')


#Frequency for Leasing based on Orders
FreqLease = Datamart['FreqOrdLease']
LeaseFreq = go.Histogram(
    x = FreqLease,
    opacity=0.75
)

OLTrace = [LeaseFreq]
layout = go.Layout(barmode='overlay', 
                   title='Client Payments for Lease', 
                   bargroupgap = 0.1,
                   xaxis = go.layout.XAxis(
                                     ticktext=['Did Not Make Lease Payments', 'Made Lease Payments'],
                                     tickvals= ['0' , '1'])
    )
PlotOL = go.Figure(data=OLTrace, layout=layout)
plotly.offline.plot(PlotOL, filename='FreqLease.html')


#Frequency for Housing based on Orders
FreqHousing = Datamart['FreqOrdHousing']

HousingFreq = go.Histogram(
    x = FreqHousing,
    opacity=0.75
)

HFTrace = [HousingFreq]
layout = go.Layout(barmode='overlay', 
                   title='Clients Payments for Household', 
                   bargroupgap = 0.1,
                   xaxis = go.layout.XAxis(
                                     ticktext=['No Payments', 'Single Payments', 'Multiple Payments'],
                                     tickvals= ['0' , '1', '2'])
    )
PlotHF = go.Figure(data=HFTrace, layout=layout)
plotly.offline.plot(PlotHF, filename='FreqHousing.html')


# Frequency for General Insurance
FreqInsurance = Datamart['FreqOrdInsurance']
InsuranceFreq = go.Histogram(
    x = FreqInsurance,
    opacity=0.75
)

IFTrace = [InsuranceFreq]
layout = go.Layout(barmode='overlay', 
                   title='Clients Payments for Insurance', 
                   bargroupgap = 0.1,
                   xaxis = go.layout.XAxis(
                                     ticktext=['No Payments', 'Payments Made'],
                                     tickvals= ['0' , '1'])
    )
PlotIF = go.Figure(data=IFTrace, layout=layout)
plotly.offline.plot(PlotIF, filename='FreqInsurance.html')


# Monetary Value for Leasing
MValLease = Datamart['MValLease']
Lease = go.Histogram(
    x = MValLease,
    opacity=0.75
)
LTrace = [Lease]
layout = go.Layout(barmode='overlay', 
                   title='Monetary Value Of Lease'
                   )
PlotL = go.Figure(data=LTrace, layout=layout)
plotly.offline.plot(PlotL, filename='MValLease.html')


# Monetary Value for Housing
MValHousing = Datamart['MValHousing']
Housing = go.Histogram(
    x = MValHousing,
    opacity=0.75
)
HTrace = [Housing]
layout = go.Layout(barmode='overlay', title='Monetary Value Of Housing')
PlotH = go.Figure(data=HTrace, layout=layout)
plotly.offline.plot(PlotH, filename='MValHousing.html')


# Monetary Value of Insurance
MValInsurance = Datamart['MValInsurance']
Insurance = go.Histogram(
    x = MValInsurance,
    opacity=0.75
)
ITrace = [Insurance]
layout = go.Layout(barmode='overlay', title='Monetary Value Of Insurance')
PlotI = go.Figure(data=ITrace, layout=layout)
plotly.offline.plot(PlotI, filename='MValInsurance.html')


# Monetary Value for Credit Transactions
MValTrnCred = Datamart['MValTrnCred']
Credit = go.Histogram(
    x = MValTrnCred,
    opacity=0.75
)
CTrace = [Credit]
layout = go.Layout(barmode='overlay', title='Monetary Value Of Credit Transactions')
PlotC = go.Figure(data=CTrace, layout=layout)
plotly.offline.plot(PlotC, filename='MValTrnCred.html')


# Monetary Value for Withdrawal Transactions
MValTrnWDR = Datamart['MValTrnWDR']
Withdraw = go.Histogram(
    x = MValTrnWDR,
    opacity=0.75
)
WDRTrace = [Withdraw]
layout = go.Layout(barmode='overlay', title = 'Monetary Value Of Withdrawal Transactions')
PlotWDR = go.Figure(data=WDRTrace, layout=layout)
plotly.offline.plot(PlotWDR, filename='MValTrnWDR.html')



# Graphs based on Loan Amount on different parameters



# Loan Amount per Gender
Datamart['AgeGroup'] = Datamart['ClientAge'] // 10 * 10
client_pv = Datamart.pivot_table(index='AgeGroup', columns='Gender', values='LoanAmount', aggfunc='sum')
AGFem = go.Bar(
    x=client_pv.index,
    y=client_pv['F'],
    name='Female'
)
AGMale = go.Bar(
    x=client_pv.index,
    y=client_pv['M'],
    name='Male'
)
AG = [AGFem, AGMale]
layout = go.Layout(
    barmode='group',
    title='Loan Amount per Age Group per Gender'
)
PlotAGL = go.Figure(data=AG, layout=layout)
plotly.offline.plot(PlotAGL, filename='AgeGroupGenderLoan.html')


#Loan Amount per Card Type
Datamart['CardGroup'] = Datamart['CardType']
client_pv = Datamart.pivot_table(index='CardGroup', columns='Gender', values='LoanAmount', aggfunc='sum')
AGFem = go.Bar(
    x=client_pv.index,
    y=client_pv['F'],
    name='Female'
)
AGMale = go.Bar(
    x=client_pv.index,
    y=client_pv['M'],
    name='Male'
)
AG = [AGFem, AGMale]
layout = go.Layout(
    barmode='group',
    title='Loan Amount per Age Group per Card Type'
)
PlotAGCL = go.Figure(data=AG, layout=layout)
plotly.offline.plot(PlotAGCL, filename='CardGroupGenderLoan.html')


# Loan Amount per Region
Datamart['RegionGroup'] = Datamart['Region']
client_pv = Datamart.pivot_table(index='RegionGroup', columns='Gender', values='LoanAmount', aggfunc='sum')
AGFem = go.Bar(
    x=client_pv.index,
    y=client_pv['F'],
    name='Female'
)
AGMale = go.Bar(
    x=client_pv.index,
    y=client_pv['M'],
    name='Male'
)
AG = [AGFem, AGMale]
layout = go.Layout(
    barmode='group',
    title='Loan Amount per Age Group per Region'
)
PlotAGRL = go.Figure(data=AG, layout=layout)
plotly.offline.plot(PlotAGRL, filename='RegionGroupLoanGender.html')

# Loan Amount per Loan Status/Type
Datamart['StatusGroup'] = Datamart['LoanStatus']
client_pv = Datamart.pivot_table(index='StatusGroup', columns='Gender', values='LoanAmount', aggfunc='sum')
AGFem = go.Bar(
    x=client_pv.index,
    y=client_pv['F'],
    name='Female'
)
AGMale = go.Bar(
    x=client_pv.index,
    y=client_pv['M'],
    name='Male'
)
AG = [AGFem, AGMale]
layout = go.Layout(
    barmode='group',
    title='Loan Amount per Age Group per Status'
)
PlotAGSL = go.Figure(data=AG, layout=layout)
plotly.offline.plot(PlotAGSL, filename='StatusGroupLoanGender.html')

# Loan Amount Based on Customer Type
Datamart['TypeGroup'] = Datamart['CustomerType']
client_pv = Datamart.pivot_table(index='TypeGroup', columns='Gender', values='LoanAmount', aggfunc='sum')
AGFem = go.Bar(
    x=client_pv.index,
    y=client_pv['F'],
    name='Female'
)
AGMale = go.Bar(
    x=client_pv.index,
    y=client_pv['M'],
    name='Male'
)
AG = [AGFem, AGMale]
layout = go.Layout(
    barmode='group',
    title='Loan Amount per Age Group per Customer Type'
)
PlotAGTL = go.Figure(data=AG, layout=layout)
plotly.offline.plot(PlotAGTL, filename='TypeGroupLoanGender.html')