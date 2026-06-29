import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import asyncio
import aiohttp
from datetime import datetime, timedelta
import time
import warnings
# numba removed - using pure numpy instead
import json
from io import StringIO
try:
    import polars as pl
    USE_POLARS = True
except ImportError:
    USE_POLARS = False
warnings.filterwarnings('ignore')

# EXPANDED STOCK LIST - Common NSE stocks
DEFAULT_STOCKS = """instrument_key,tradingsymbol
NSE_EQ|INE585B01010,MARUTI
NSE_EQ|INE139A01034,NATIONALUM
NSE_EQ|INE763I01026,TARIL
NSE_EQ|INE970X01018,LEMONTREE
NSE_EQ|INE522D01027,MANAPPURAM
NSE_EQ|INE427F01016,CHALET
NSE_EQ|INE00R701025,DALBHARAT
NSE_EQ|INE917I01010,BAJAJ-AUTO
NSE_EQ|INE146L01010,KIRLOSENG
NSE_EQ|INE267A01025,HINDZINC
NSE_EQ|INE466L01038,360ONE
NSE_EQ|INE070A01015,SHREECEM
NSE_EQ|INE242C01024,ANANTRAJ
NSE_EQ|INE883F01010,AADHARHFC
NSE_EQ|INE749A01030,JINDALSTEL
NSE_EQ|INE171Z01026,BDL
NSE_EQ|INE591G01017,COFORGE
NSE_EQ|INE903U01023,SIGNATURE
NSE_EQ|INE160A01022,PNB
NSE_EQ|INE640A01023,SKFINDIA
NSE_EQ|INE814H01011,ADANIPOWER
NSE_EQ|INE736A01011,CDSL
NSE_EQ|INE301A01014,RAYMOND
NSE_EQ|INE102D01028,GODREJCP
NSE_EQ|INE600L01024,LALPATHLAB
NSE_EQ|INE134E01011,PFC
NSE_EQ|INE269A01021,SONATSOFTW
NSE_EQ|INE009A01021,INFY
NSE_EQ|INE962Y01021,IRCON
NSE_EQ|INE048G01026,NAVINFLUOR
NSE_EQ|INE918Z01012,KAYNES
NSE_EQ|INE376G01013,BIOCON
NSE_EQ|INE00M201021,SWSOLAR
NSE_EQ|INE619A01035,PATANJALI
NSE_EQ|INE465A01025,BHARATFORG
NSE_EQ|INE589A01014,NLCINDIA
NSE_EQ|INE463A01038,BERGEPAINT
NSE_EQ|INE622W01025,ACMESOLAR
NSE_EQ|INE256A01028,ZEEL
NSE_EQ|INE540L01014,ALKEM
NSE_EQ|INE237A01028,KOTAKBANK
NSE_EQ|INE126A01031,EIDPARRY
NSE_EQ|INE482A01020,CEATLTD
NSE_EQ|INE850D01014,GODREJAGRO
NSE_EQ|INE361B01024,DIVISLAB
NSE_EQ|INE517B01013,TTML
NSE_EQ|INE385C01021,SARDAEN
NSE_EQ|INE811K01011,PRESTIGE
NSE_EQ|INE01EA01019,VMM
NSE_EQ|INE510A01028,ENGINERSIN
NSE_EQ|INE030A01027,HINDUNILVR
NSE_EQ|INE872J01023,DEVYANI
NSE_EQ|INE476A01022,CANBK
NSE_EQ|INE419U01012,HAPPSTMNDS
NSE_EQ|INE691A01018,UCOBANK
NSE_EQ|INE745G01035,MCX
NSE_EQ|INE0W2G01015,SAGILITY
NSE_EQ|INE531E01026,HINDCOPPER
NSE_EQ|INE483C01032,TANLA
NSE_EQ|INE721A01047,SHRIRAMFIN
NSE_EQ|INE028A01039,BANKBARODA
NSE_EQ|INE670K01029,LODHA
NSE_EQ|INE039A01010,IFCI
NSE_EQ|INE914M01019,ASTERDM
NSE_EQ|INE570L01029,SAILIFE
NSE_EQ|INE158A01026,HEROMOTOCO
NSE_EQ|INE112L01020,METROPOLIS
NSE_EQ|INE405E01023,UNOMINDA
NSE_EQ|INE777K01022,RRKABEL
NSE_EQ|INE123W01016,SBILIFE
NSE_EQ|INE192A01025,TATACONSUM
NSE_EQ|INE398R01022,SYNGENE
NSE_EQ|INE118A01012,BAJAJHLDNG
NSE_EQ|INE371A01025,GRAPHITE
NSE_EQ|INE373A01013,BASF
NSE_EQ|INE674K01013,ABCAPITAL
NSE_EQ|INE094A01015,HINDPETRO
NSE_EQ|INE410P01011,NH
NSE_EQ|INE203A01020,ASTRAZEN
NSE_EQ|INE528G01035,YESBANK
NSE_EQ|INE248A01017,ITI
NSE_EQ|INE531F01015,NUVAMA
NSE_EQ|INE093I01010,OBEROIRLTY
NSE_EQ|INE616N01034,INOXINDIA
NSE_EQ|INE726G01019,ICICIPRULI
NSE_EQ|INE901L01018,APLLTD
NSE_EQ|INE271B01025,MAHSEAMLES
NSE_EQ|INE073K01018,SONACOMS
NSE_EQ|INE006I01046,ASTRAL
NSE_EQ|INE142M01025,TATATECH
NSE_EQ|INE036D01028,KARURVYSYA
NSE_EQ|INE885A01032,ARE&M
NSE_EQ|INE233B01017,BLUEDART
NSE_EQ|INE169A01031,COROMANDEL
NSE_EQ|INE235A01022,FINCABLES
NSE_EQ|INE668F01031,JYOTHYLAB
NSE_EQ|INE849A01020,TRENT
NSE_EQ|INE669C01036,TECHM
NSE_EQ|INE322A01010,GILLETTE
NSE_EQ|INE216A01030,BRITANNIA
NSE_EQ|INE002S01010,MGL
NSE_EQ|INE111A01025,CONCOR
NSE_EQ|INE531A01024,KANSAINER
NSE_EQ|INE062A01020,SBIN
NSE_EQ|INE180C01042,CGCL
NSE_EQ|INE128S01021,FIVESTAR
NSE_EQ|INE672A01018,TATAINVEST
NSE_EQ|INE216P01012,AAVAS
NSE_EQ|INE220B01022,KPIL
NSE_EQ|INE081A01020,TATASTEEL
NSE_EQ|INE007A01025,CRISIL
NSE_EQ|INE883A01011,MRF
NSE_EQ|INE824G01012,JSWHL
NSE_EQ|INE075A01022,WIPRO
NSE_EQ|INE498L01015,LTF
NSE_EQ|INE377N01017,WAAREEENER
NSE_EQ|INE484J01027,GODREJPROP
NSE_EQ|INE979A01025,SAREGAMA
NSE_EQ|INE188A01015,FACT
NSE_EQ|INE205A01025,VEDL
NSE_EQ|INE027H01010,MAXHEALTH
NSE_EQ|INE298J01013,NAM-INDIA
NSE_EQ|INE101D01020,GRANULES
NSE_EQ|INE212H01026,AIAENG
NSE_EQ|INE967H01025,KIMS
NSE_EQ|INE121A01024,CHOLAFIN
NSE_EQ|INE010J01012,TEJASNET
NSE_EQ|INE474Q01031,MEDANTA
NSE_EQ|INE839M01018,SCHNEIDER
NSE_EQ|INE074A01025,PRAJIND
NSE_EQ|INE974X01010,TIINDIA
NSE_EQ|INE854D01024,UNITDSPR
NSE_EQ|INE220G01021,JSL
NSE_EQ|INE742F01042,ADANIPORTS
NSE_EQ|INE226A01021,VOLTAS
NSE_EQ|INE0NT901020,NETWEB
NSE_EQ|INE292B01021,HBLENGINE
NSE_EQ|INE047A01021,GRASIM
NSE_EQ|INE326A01037,LUPIN
NSE_EQ|INE584A01023,NMDC
NSE_EQ|INE085A01013,CHAMBLFERT
NSE_EQ|INE03Q201024,ALIVUS
NSE_EQ|INE836A01035,BSOFT
NSE_EQ|INE548A01028,HFCL
NSE_EQ|INE501A01019,DEEPAKFERT
NSE_EQ|INE414G01012,MUTHOOTFIN
NSE_EQ|INE669E01016,IDEA
NSE_EQ|INE743M01012,RHIM
NSE_EQ|INE324A01032,JINDALSAW
NSE_EQ|INE211B01039,PHOENIXLTD
NSE_EQ|INE813H01021,TORNTPOWER
NSE_EQ|INE066P01011,INOXWIND
NSE_EQ|INE880J01026,JSWINFRA
NSE_EQ|INE358A01014,ABBOTINDIA
NSE_EQ|INE868B01028,NCC
NSE_EQ|INE172A01027,CASTROLIND
NSE_EQ|INE213A01029,ONGC
NSE_EQ|INE825A01020,VTL
NSE_EQ|INE0FS801015,MSUMI
NSE_EQ|INE335Y01020,IRCTC
NSE_EQ|INE406M01024,ERIS
NSE_EQ|INE725A01030,NAVA
NSE_EQ|INE00WC01027,AFFLE
NSE_EQ|INE931S01010,ADANIENSOL
NSE_EQ|INE704P01025,COCHINSHIP
NSE_EQ|INE053F01010,IRFC
NSE_EQ|INE127D01025,HDFCAMC
NSE_EQ|INE021A01026,ASIANPAINT
NSE_EQ|INE671A01010,HONAUT
NSE_EQ|INE356A01018,MPHASIS
NSE_EQ|INE571A01038,IPCALAB
NSE_EQ|INE733E01010,NTPC
NSE_EQ|INE230A01023,EIHOTEL
NSE_EQ|INE565A01014,IOB
NSE_EQ|INE022Q01020,IEX
NSE_EQ|INE115A01026,LICHSGFIN
NSE_EQ|INE475E01026,CAPLIPOINT
NSE_EQ|INE463V01026,ANANDRATHI
NSE_EQ|INE596I01012,CAMS
NSE_EQ|INE684F01012,FSL
NSE_EQ|INE702C01027,APLAPOLLO
NSE_EQ|INE017A01032,GESHIP
NSE_EQ|INE388Y01029,NYKAA
NSE_EQ|INE348B01021,CENTURYPLY
NSE_EQ|INE117A01022,ABB
NSE_EQ|INE239A01024,NESTLEIND
NSE_EQ|INE02ID01020,RAYMONDLSL
NSE_EQ|INE980O01024,JYOTICNC
NSE_EQ|INE228A01035,USHAMART
NSE_EQ|INE437A01024,APOLLOHOSP
NSE_EQ|INE245A01021,TATAPOWER
NSE_EQ|INE288B01029,DEEPAKNTR
NSE_EQ|INE053A01029,INDHOTEL
NSE_EQ|INE927D01051,JBMA
NSE_EQ|INE995S01015,NIVABUPA
NSE_EQ|INE100A01010,ATUL
NSE_EQ|INE665A01038,SWANENERGY
NSE_EQ|INE196A01026,MARICO
NSE_EQ|INE338H01029,CONCORDBIO
NSE_EQ|INE152M01016,TRITURBINE
NSE_EQ|INE121J01017,INDUSTOWER
NSE_EQ|INE140A01024,PEL
NSE_EQ|INE389H01022,KEC
NSE_EQ|INE399L01023,ATGL
NSE_EQ|INE055A01016,ABREL
NSE_EQ|INE024L01027,GRAVITA
NSE_EQ|INE615H01020,TITAGARH
NSE_EQ|INE121E01018,JSWENERGY
NSE_EQ|INE019A01038,JSWSTEEL
NSE_EQ|INE0IX101010,DATAPATTNS
NSE_EQ|INE450U01017,ROUTE
NSE_EQ|INE151A01013,TATACOMM
NSE_EQ|INE522F01014,COALINDIA
NSE_EQ|INE382Z01011,GRSE
NSE_EQ|INE095N01031,NBCC
NSE_EQ|INE296A01024,BAJFINANCE
NSE_EQ|INE066F01020,HAL
NSE_EQ|INE002A01018,RELIANCE
NSE_EQ|INE462A01022,BAYERCROP
NSE_EQ|INE961O01016,RAINBOW
NSE_EQ|INE203G01027,IGL
NSE_EQ|INE619B01017,NEWGEN
NSE_EQ|INE109A01011,SCI
NSE_EQ|INE183A01024,FINPIPE
NSE_EQ|INE113A01013,GNFC
NSE_EQ|INE467B01029,TCS
NSE_EQ|INE573A01042,JKTYRE
NSE_EQ|INE806T01020,SAPPHIRE
NSE_EQ|INE473A01011,LINDEINDIA
NSE_EQ|INE153T01027,BLS
NSE_EQ|INE258A01016,BEML
NSE_EQ|INE759A01021,MASTEK
NSE_EQ|INE0ONG01011,NTPCGREEN
NSE_EQ|INE149A01033,CHOLAHLDNG
NSE_EQ|INE192B01031,WELSPUNLIV
NSE_EQ|INE079A01024,AMBUJACEM
NSE_EQ|INE457L01029,PGEL
NSE_EQ|INE0J1Y01017,LICI
NSE_EQ|INE260B01028,GODFRYPHLP
NSE_EQ|INE299U01018,CROMPTON
NSE_EQ|INE040A01034,HDFCBANK
NSE_EQ|INE200A01026,GVT&D
NSE_EQ|INE121A08PJ0,CHOLAFIN
NSE_EQ|INE270A01029,ALOKINDS
NSE_EQ|INE371P01015,AMBER
NSE_EQ|INE205B01031,ELECON
NSE_EQ|INE486A01021,CESC
NSE_EQ|INE399G01023,RKFORGE
NSE_EQ|INE603J01030,PIIND
NSE_EQ|INE202E01016,IREDA
NSE_EQ|INE663F01032,NAUKRI
NSE_EQ|INE066A01021,EICHERMOT
NSE_EQ|INE844O01030,GUJGASLTD
NSE_EQ|INE481N01025,HOMEFIRST
NSE_EQ|INE421D01022,CCL
NSE_EQ|INE752E01010,POWERGRID
NSE_EQ|INE271C01023,DLF
NSE_EQ|INE318A01026,PIDILITIND
NSE_EQ|INE208C01025,AEGISLOG
NSE_EQ|INE520A01027,ZENSARTECH
NSE_EQ|INE818H01020,LTFOODS
NSE_EQ|INE499A01024,DCMSHRIRAM
NSE_EQ|INE306R01017,INTELLECT
NSE_EQ|INE042A01014,ESCORTS
NSE_EQ|INE176A01028,BATAINDIA
NSE_EQ|INE064C01022,TRIDENT
NSE_EQ|INE285K01026,TECHNOE
NSE_EQ|INE256C01024,TRIVENI
NSE_EQ|INE274F01020,WESTLIFE
NSE_EQ|INE947Q01028,LAURUSLABS
NSE_EQ|INE913H01037,ENDURANCE
NSE_EQ|INE918I01026,BAJAJFINSV
NSE_EQ|INE758E01017,JIOFIN
NSE_EQ|INE089A01031,DRREDDY
NSE_EQ|INE251B01027,ZENTEC
NSE_EQ|INE575P01011,STARHEALTH
NSE_EQ|INE195J01029,PNCINFRA
NSE_EQ|INE834M01019,RTNINDIA
NSE_EQ|INE848E01016,NHPC
NSE_EQ|INE852O01025,APTUS
NSE_EQ|INE545A01024,HEG
NSE_EQ|INE982J01020,PAYTM
NSE_EQ|INE205C01021,POLYMED
NSE_EQ|INE634I01029,KNRCON
NSE_EQ|INE761H01022,PAGEIND
NSE_EQ|INE342J01019,ZFCVINDIA
NSE_EQ|INE494B01023,TVSMOTOR
NSE_EQ|INE673O01025,TBOTEK
NSE_EQ|INE646L01027,INDIGO
NSE_EQ|INE0V6F01027,HYUNDAI
NSE_EQ|INE010B01027,ZYDUSLIFE
NSE_EQ|INE302A01020,EXIDEIND
NSE_EQ|INE0BY001018,JUBLINGREA
NSE_EQ|INE810G01011,SHYAMMETL
NSE_EQ|INE351F01018,JPPOWER
NSE_EQ|INE634S01028,MANKIND
NSE_EQ|INE191B01025,WELCORP
NSE_EQ|INE397D01024,BHARTIARTL
NSE_EQ|INE192R01011,DMART
NSE_EQ|INE686F01025,UBL
NSE_EQ|INE123F01029,MMTC
NSE_EQ|INE008A01015,IDBI
NSE_EQ|INE321T01012,DOMS
NSE_EQ|INE775A08105,MOTHERSON
NSE_EQ|INE933S01016,INDIAMART
NSE_EQ|INE732I01013,ANGELONE
NSE_EQ|INE059A01026,CIPLA
NSE_EQ|INE00E101023,BIKAJI
NSE_EQ|INE660A01013,SUNDARMFIN
NSE_EQ|INE03QK01018,COHANCE
NSE_EQ|INE138Y01010,KFINTECH
NSE_EQ|INE377Y01014,BAJAJHFL
NSE_EQ|INE168P01015,EMCURE
NSE_EQ|INE343G01021,BHARTIHEXA
NSE_EQ|INE481Y01014,GICRE
NSE_EQ|INE797F01020,JUBLFOOD
NSE_EQ|INE180A01020,MFSL
NSE_EQ|INE949L01017,AUBANK
NSE_EQ|INE881D01027,OFSS
NSE_EQ|INE795G01014,HDFCLIFE
NSE_EQ|INE439A01020,ASAHIINDIA
NSE_EQ|INE148I01020,SAMMAANCAP
NSE_EQ|INE823G01014,JKCEMENT
NSE_EQ|INE987B01026,NATCOPHARM
NSE_EQ|INE280A01028,TITAN
NSE_EQ|INE227W01023,CLEAN
NSE_EQ|INE716A01013,WHIRLPOOL
NSE_EQ|INE03JT01014,GODIGIT
NSE_EQ|INE298A01020,CUMMINSIND
NSE_EQ|INE470Y01017,NIACL
NSE_EQ|INE769A01020,AARTIIND
NSE_EQ|INE155A01022,TATAMOTORS
NSE_EQ|INE119A01028,BALRAMCHIN
NSE_EQ|INE258G01013,SUMICHEM
NSE_EQ|INE930H01031,KPRMILL
NSE_EQ|INE614G01033,RPOWER
NSE_EQ|INE274J01014,OIL
NSE_EQ|INE372A01015,APARINDS
NSE_EQ|INE02RE01045,FIRSTCRY
NSE_EQ|INE285A01027,ELGIEQUIP
NSE_EQ|INE383A01012,INDIACEM
NSE_EQ|INE012A01025,ACC
NSE_EQ|INE0NNS01018,NSLNISP
NSE_EQ|INE944F01028,RADICO
NSE_EQ|INE572E01012,PNBHOUSING
NSE_EQ|INE281B01032,LLOYDSME
NSE_EQ|INE050A01025,BBTC
NSE_EQ|INE095A01012,INDUSINDBK
NSE_EQ|INE09N301011,FLUOROCHEM
NSE_EQ|INE513A01022,SCHAEFFLER
NSE_EQ|INE562A01011,INDIANB
NSE_EQ|INE780C01023,JMFINANCIL
NSE_EQ|INE195A01028,SUPREMEIND
NSE_EQ|INE049B01025,WOCKPHARMA
NSE_EQ|INE483A01010,CENTRALBK
NSE_EQ|INE136B01020,CYIENT
NSE_EQ|INE043W01024,VIJAYA
NSE_EQ|INE209L01016,JWL
NSE_EQ|INE168A01041,J&KBANK
NSE_EQ|INE870H01013,NETWORK18
NSE_EQ|INE118H01025,BSE
NSE_EQ|INE364U01010,ADANIGREEN
NSE_EQ|INE101I01011,AFCONS
NSE_EQ|INE238A01034,AXISBANK
NSE_EQ|INE065X01017,INDGN
NSE_EQ|INE044A01036,SUNPHARMA
NSE_EQ|INE177H01039,GPIL
NSE_EQ|INE470A01017,3MINDIA
NSE_EQ|INE338I01027,MOTILALOFS
NSE_EQ|INE935N01020,DIXON
NSE_EQ|INE002L01015,SJVN
NSE_EQ|INE038A01020,HINDALCO
NSE_EQ|INE031A01017,HUDCO
NSE_EQ|INE027A01015,RCF
NSE_EQ|INE242A01010,IOC
NSE_EQ|INE0DK501011,PPLPHARMA
NSE_EQ|INE0BV301023,MAPMYINDIA
NSE_EQ|INE131A01031,GMDCLTD
NSE_EQ|INE692A01016,UNIONBANK
NSE_EQ|INE477A01020,CANFINHOME
NSE_EQ|INE739E01017,CERA
NSE_EQ|INE04I401011,KPITTECH
NSE_EQ|INE061F01013,FORTIS
NSE_EQ|INE010V01017,LTTS
NSE_EQ|INE263A01024,BEL
NSE_EQ|INE120A01034,CARBORUNIV
NSE_EQ|INE020B01018,RECLTD
NSE_EQ|INE685A01028,TORNTPHARM
NSE_EQ|INE647A01010,SRF
NSE_EQ|INE491A01021,CUB
NSE_EQ|INE517F01014,GPPL
NSE_EQ|INE860A01027,HCLTECH
NSE_EQ|INE0BS701011,PREMIERENE
NSE_EQ|INE00H001014,SWIGGY
NSE_EQ|INE178A01016,CHENNPETRO
NSE_EQ|INE457A01014,MAHABANK
NSE_EQ|INE891D01026,REDINGTON
NSE_EQ|INE671H01015,SOBHA
NSE_EQ|INE278Y01022,CAMPUS
NSE_EQ|INE171A01029,FEDERALBNK
NSE_EQ|INE976G01028,RBLBANK
NSE_EQ|INE262H01021,PERSISTENT
NSE_EQ|INE084A01016,BANKINDIA
NSE_EQ|INE775A01035,MOTHERSON
NSE_EQ|INE217B01036,KAJARIACER
NSE_EQ|INE878B01027,KEI
NSE_EQ|INE599M01018,JUSTDIAL
NSE_EQ|INE325A01013,TIMKEN
NSE_EQ|INE741K01010,CREDITACC
NSE_EQ|INE018E01016,SBICARD
NSE_EQ|INE0LXG01040,OLAELEC
NSE_EQ|INE776C01039,GMRAIRPORT
NSE_EQ|INE417T01026,POLICYBZR
NSE_EQ|INE068V01023,GLAND
NSE_EQ|INE115Q01022,IKS
NSE_EQ|INE602A01031,PCBL
NSE_EQ|INE879I01012,DBREALTY
NSE_EQ|INE415G01027,RVNL
NSE_EQ|INE791I01019,BRIGADE
NSE_EQ|INE821I01022,IRB
NSE_EQ|INE323A01026,BOSCHLTD
NSE_EQ|INE320J01015,RITES
NSE_EQ|INE182A01018,PFIZER
NSE_EQ|INE548C01032,EMAMILTD
NSE_EQ|INE214T01019,LTIM
NSE_EQ|INE176B01034,HAVELLS
NSE_EQ|INE404A01024,ABSLAMC
NSE_EQ|INE545U01014,BANDHANBNK
NSE_EQ|INE152A01029,THERMAX
NSE_EQ|INE511C01022,POONAWALLA
NSE_EQ|INE150B01039,ALKYLAMINE
NSE_EQ|INE249Z01020,MAZDOCK
NSE_EQ|INE0DD101019,RAILTEL
NSE_EQ|INE087H01022,RENUKA
NSE_EQ|INE343H01029,SOLARINDS
NSE_EQ|INE732A01036,KIRLOSBROS
NSE_EQ|INE191H01014,PVRINOX
NSE_EQ|INE094J01016,UTIAMC
NSE_EQ|INE530B01024,IIFL
NSE_EQ|INE758T01015,ETERNAL
NSE_EQ|INE154A01025,ITC
NSE_EQ|INE455K01017,POLYCAB
NSE_EQ|INE406A01037,AUROPHARMA
NSE_EQ|INE387A01021,SUNDRMFAST
NSE_EQ|INE101A01026,M&M
NSE_EQ|INE208A01029,ASHOKLEY
NSE_EQ|INE303R01014,KALYANKJIL
NSE_EQ|INE148O01028,DELHIVERY
NSE_EQ|INE331A01037,RAMCOCEM
NSE_EQ|INE090A01021,ICICIBANK
NSE_EQ|INE472A01039,BLUESTARCO
NSE_EQ|INE628A01036,UPL
NSE_EQ|INE159A01016,GLAXO
NSE_EQ|INE787D01026,BALKRISIND
NSE_EQ|INE040H01021,SUZLON
NSE_EQ|INE09XN01023,AKUMS
NSE_EQ|INE018A01030,LT
NSE_EQ|INE092T01019,IDFCFIRSTB
NSE_EQ|INE700A01033,JUBLPHARMA
NSE_EQ|INE347G01014,PETRONET
NSE_EQ|INE103A01014,MRPL
NSE_EQ|INE067A01029,CGPOWER
NSE_EQ|INE438A01022,APOLLOTYRE
NSE_EQ|INE260D01016,OLECTRA
NSE_EQ|INE794A01010,NEULANDLAB
NSE_EQ|INE423A01024,ADANIENT
NSE_EQ|INE259A01022,COLPAL
NSE_EQ|INE07Y701011,POWERINDIA
NSE_EQ|INE765G01017,ICICIGI
NSE_EQ|INE257A01026,BHEL
NSE_EQ|INE774D01024,M&MFIN
NSE_EQ|INE206F01022,AIIL
NSE_EQ|INE424H01027,SUNTV
NSE_EQ|INE842C01021,MINDACORP
NSE_EQ|INE246F01010,GSPL
NSE_EQ|INE699H01024,AWL
NSE_EQ|INE647O01011,ABFRL
NSE_EQ|INE019C01026,HSCL
NSE_EQ|INE129A01019,GAIL
NSE_EQ|INE825V01034,MANYAVAR
NSE_EQ|INE731H01025,ACE
NSE_EQ|INE423Y01016,SBFC
NSE_EQ|INE481G01011,ULTRACEMCO
NSE_EQ|INE572A01036,JBCHEPHARM
NSE_EQ|INE0I7C01011,LATENTVIEW
NSE_EQ|INE233A01035,GODREJIND
NSE_EQ|INE114A01011,SAIL
NSE_EQ|INE031B01049,AJANTPHARM
NSE_EQ|INE774D08MG3,M&MFIN
NSE_EQ|INE935A01035,GLENMARK
NSE_EQ|INE003A01024,SIEMENS
NSE_EQ|INE029A01011,BPCL
NSE_EQ|INE670A01012,TATAELXSI
NSE_EQ|INE951I01027,VGUARD
NSE_EQ|INE092A01019,TATACHEM
NSE_EQ|INE200M01039,VBL
NSE_EQ|INE0DYJ01015,SYRMA
NSE_EQ|INE738I01010,ECLERX
NSE_EQ|INE00LO01017,CRAFTSMAN
NSE_EQ|INE0J5401028,HONASA
NSE_EQ|INE0Q9301021,IGIL
NSE_EQ|INE016A01026,DABUR
NSE_EQ|INE596F01018,PTCIL"""

# ─────────────────────────────────────────────
# WhatsApp Configuration (Green API - Free)
# ─────────────────────────────────────────────
# SETUP STEPS (one-time, 5 minutes):
#
# 1. Go to https://green-api.com and click "Try Free"
# 2. Register with your email
# 3. Create a new Instance → you get:
#      INSTANCE_ID  (e.g. 1101234567)
#      API_TOKEN    (e.g. abcdef1234567890abcdef)
# 4. In the dashboard, click "Scan QR" and scan with
#    YOUR WhatsApp (the number that will SEND messages)
# 5. For GROUP chat ID:
#      Open WhatsApp Web → open the group
#      The URL will show: .../#/120363XXXXXXXXXX@g.us
#      Your GROUP_CHAT_ID = "120363XXXXXXXXXX@g.us"
# 6. For PERSONAL number:
#      PERSONAL_CHAT_ID = "919876543210@c.us"
#      (91 = country code, then mobile number, no + or spaces)
# ─────────────────────────────────────────────

GREEN_API_INSTANCE_ID = "710701668156"       # e.g. "1101234567"
GREEN_API_TOKEN       = "4c7003de664449d496f032a4ce98773cfa9e85b9c2e9425d9d"       # e.g. "abcdef1234567890abcdef"

# Fill one or both targets:
WHATSAPP_PERSONAL_CHAT_ID = ""  # e.g. "919876543210@c.us"
WHATSAPP_GROUP_CHAT_ID    = "120363410390351067@g.us"  # e.g. "120363XXXXXXXXXX@g.us"

async def _green_api_send(session: aiohttp.ClientSession, chat_id: str, message: str, label: str):
    """Send one WhatsApp message via Green API"""
    url = (f"https://api.green-api.com/waInstance{GREEN_API_INSTANCE_ID}"
           f"/sendMessage/{GREEN_API_TOKEN}")
    payload = {"chatId": chat_id, "message": message}
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json()
            if resp.status == 200 and data.get("idMessage"):
                return True, label
            else:
                return False, f"{label}: {data}"
    except Exception as e:
        return False, f"{label}: {str(e)[:100]}"

async def send_whatsapp_message(message: str):
    """Send WhatsApp message to personal and/or group via Green API"""
    if not GREEN_API_INSTANCE_ID or not GREEN_API_TOKEN:
        return [(False, "Green API credentials not set. Fill GREEN_API_INSTANCE_ID and GREEN_API_TOKEN.")]

    targets = []
    if WHATSAPP_PERSONAL_CHAT_ID:
        targets.append((WHATSAPP_PERSONAL_CHAT_ID, "Personal"))
    if WHATSAPP_GROUP_CHAT_ID:
        targets.append((WHATSAPP_GROUP_CHAT_ID, "Group"))

    if not targets:
        return [(False, "No chat IDs configured. Fill WHATSAPP_PERSONAL_CHAT_ID or WHATSAPP_GROUP_CHAT_ID.")]

    async with aiohttp.ClientSession() as session:
        tasks = [_green_api_send(session, chat_id, message, label)
                 for chat_id, label in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return [(r if isinstance(r, tuple) else (False, str(r))) for r in results]

# Page configuration
st.set_page_config(page_title="Upstox Scanner Ultra", page_icon="🚀", layout="wide")

# Custom CSS
st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:bold;color:#1f77b4;text-align:center;margin-bottom:2rem;}
.metric-card{background-color:#f0f2f6;padding:1rem;border-radius:0.5rem;margin:0.5rem 0;}
.upload-section{background-color:#f8f9fa;padding:1rem;border-radius:0.5rem;margin:1rem 0;}
</style>""", unsafe_allow_html=True)

# Cache for historical data
@st.cache_data(ttl=600)
def get_cached_historical_data(symbol, period="1y"):
    try:
        ticker = f"{symbol}.NS"
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None
        # Trim to last 300 candles for optimization
        return data.tail(300)
    except:
        return None

# Pure numpy calculations (numba removed for Python 3.13 compatibility)
def fast_ema_calculation(prices, period):
    prices = np.asarray(prices, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

def fast_rsi_calculation(prices, period=14):
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
    avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    result = np.full(len(prices), 50.0)
    result[period:] = rsi
    return result

def fast_stochrsi_calculation(rsi, period=14):
    rsi = np.asarray(rsi, dtype=np.float64)
    if len(rsi) < period:
        return np.full(len(rsi), 50.0)
    stochrsi = np.full(len(rsi), 50.0)
    for i in range(period-1, len(rsi)):
        rsi_slice = rsi[i-period+1:i+1]
        min_rsi = np.min(rsi_slice)
        max_rsi = np.max(rsi_slice)
        if max_rsi - min_rsi != 0:
            stochrsi[i] = 100 * (rsi[i] - min_rsi) / (max_rsi - min_rsi)
        else:
            stochrsi[i] = 50.0
    return stochrsi

def fast_atr_calculation(high, low, close, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if len(high) < period:
        return np.full(len(high), 0.5)
    tr = np.empty(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.convolve(tr, np.ones(period)/period, mode='valid')
    result = np.full(len(high), atr[0] if len(atr) > 0 else 0.5)
    result[period-1:] = atr
    return result

def fast_sma_calculation(prices, period):
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < period:
        return np.full(len(prices), np.mean(prices))
    sma = np.convolve(prices, np.ones(period)/period, mode='valid')
    result = np.full(len(prices), sma[0] if len(sma) > 0 else prices[0])
    result[period-1:] = sma
    return result

def calculate_indicators_ultra_fast(data):
    if data is None or len(data) < 50:
        return None
    df = data.copy()
    prices = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    open_prices = df['Open'].values
    
    # Calculate all indicators once
    df['EMA5'] = fast_ema_calculation(prices, 5)
    df['EMA12'] = fast_ema_calculation(prices, 12)
    df['EMA13'] = fast_ema_calculation(prices, 13)
    df['EMA26'] = fast_ema_calculation(prices, 26)
    df['SMA20'] = fast_sma_calculation(prices, 20)
    df['SMA50'] = fast_sma_calculation(prices, 50)
    df['SMA100'] = fast_sma_calculation(prices, 100)
    df['SMA200'] = fast_sma_calculation(prices, 200)
    
    # RSI and StochRSI
    rsi = fast_rsi_calculation(prices)
    df['RSI'] = rsi
    df['StochRSI'] = fast_stochrsi_calculation(rsi)
    
    # ATR
    df['ATR'] = fast_atr_calculation(high, low, prices)
    
    # MACD (reuse EMA12 and EMA26)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = fast_ema_calculation(df['MACD'].values, 9)
    
    # Bollinger Bands (reuse SMA20)
    bb_std = pd.Series(prices).rolling(20).std().fillna(0).values
    df['BB_Upper'] = df['SMA20'] + (bb_std * 2)
    
    # Volume and high calculations
    df['Volume_SMA50'] = fast_sma_calculation(volume, 50)
    df['High_200'] = pd.Series(high).rolling(min(200, len(df))).max().fillna(high[-1]).values
    df['High_52w'] = pd.Series(high).rolling(min(252, len(df))).max().fillna(high[-1]).values
    
    return df

def check_conditions_vectorized(price, ema5, ema13, ema26, sma50, sma100, sma200, rsi, stochrsi, 
                               macd, macd_signal, volume, vol_sma50, atr, bb_upper, high_200, 
                               high_52w, open_price, low, prev_high, prev_close, high_current, weights):
    """Vectorized condition checking with NumPy operations"""
    conditions = np.zeros(16, dtype=np.bool_)
    
    # All 15 original conditions + 1 new StochRSI condition
    conditions[0] = price > ema5 > ema13 > ema26 > 0
    conditions[1] = price > sma50 > sma100 > sma200 > 0
    conditions[2] = rsi > 55
    conditions[3] = stochrsi > 50  # New StochRSI condition
    conditions[4] = macd > macd_signal
    conditions[5] = volume > 100000 and volume > vol_sma50
    conditions[6] = price > open_price
    conditions[7] = price >= bb_upper
    conditions[8] = price > (1.05 * high_200)
    conditions[9] = low > prev_high
    conditions[10] = price >= (0.97 * high_current)
    conditions[11] = price >= (0.95 * high_52w)
    conditions[12] = (atr / price) < 0.06
    conditions[13] = volume > (2 * vol_sma50) and (price - prev_close) / prev_close > 0.02
    conditions[14] = price > prev_close * 1.01
    conditions[15] = volume > vol_sma50 * 1.5
    
    # Calculate weighted score
    score = np.sum(conditions.astype(np.float64) * weights)
    total_conditions = np.sum(conditions)
    
    return total_conditions, score

# Async quote fetching
async def fetch_batch_quotes(session, keys, token, batch_id, semaphore):
    url = "https://api.upstox.com/v2/market-quote/quotes"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}'}
    params = {'instrument_key': ','.join(keys)}
    
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, headers=headers, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', {}), batch_id
                    elif response.status == 429:
                        await asyncio.sleep(0.05 * (2 ** attempt))
                    else:
                        break
            except Exception as e:
                if attempt == 2:
                    print(f"Batch {batch_id} failed: {str(e)[:50]}...")
                await asyncio.sleep(0.1)
    return {}, batch_id

async def get_all_quotes_async(instrument_keys, token, batch_size=500, max_concurrent=20):
    """Async quote fetching with rate limiting"""
    quotes = {}
    batches = [instrument_keys[i:i+batch_size] for i in range(0, len(instrument_keys), batch_size)]
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_batch_quotes(session, batch, token, i, semaphore) 
                for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, tuple):
                batch_quotes, batch_id = result
                quotes.update(batch_quotes)
    
    return quotes

def safe_val(val, default=0):
    try:
        return float(val) if pd.notna(val) and np.isfinite(val) else default
    except:
        return default

def process_single_stock_optimized(args):
    symbol, price, volume, min_conditions, weights = args
    try:
        hist_data = get_cached_historical_data(symbol)
        if hist_data is None:
            return None
        
        hist_data = calculate_indicators_ultra_fast(hist_data)
        if hist_data is None:
            return None
        
        latest = hist_data.iloc[-1]
        prev = hist_data.iloc[-2] if len(hist_data) > 1 else latest
        
        # Extract values
        values = [
            price, safe_val(latest['EMA5']), safe_val(latest['EMA13']), safe_val(latest['EMA26']),
            safe_val(latest['SMA50']), safe_val(latest['SMA100']), safe_val(latest['SMA200']),
            safe_val(latest['RSI'], 50), safe_val(latest['StochRSI'], 50),
            safe_val(latest['MACD']), safe_val(latest['MACD_Signal']), volume,
            safe_val(latest['Volume_SMA50']), safe_val(latest['ATR'], 0.5),
            safe_val(latest['BB_Upper'], price), safe_val(latest['High_200']),
            safe_val(latest['High_52w']), safe_val(latest['Open']), safe_val(latest['Low']),
            safe_val(prev['High']), safe_val(prev['Close']), safe_val(latest['High'])
        ]
        
        conditions, score = check_conditions_vectorized(*values, weights)
        
        if conditions >= min_conditions:
            atr = values[13]
            return {
                'Symbol': symbol,
                'CMP (₹)': round(price, 2),
                'Volume': int(volume),
                'ATR (₹)': round(atr, 2),
                'Target 1 (₹)': round(price + (1.5 * atr), 2),
                'Target 2 (₹)': round(price + (2.0 * atr), 2),
                'Stoploss (₹)': round(price - (1.0 * atr), 2),
                'Conditions': f"{conditions}/16",
                'Score': round(score, 2),
                'Entry_Date': datetime.today().date()
            }
    except:
        pass
    return None

def format_telegram_message(qualifying_stocks):
    """Format results for Telegram with premium Elite Swing Scanner layout."""
    if not qualifying_stocks:
        return "📊 *UPSTOX ELITE SWING SCANNER*\n\n❌ No stocks found matching criteria"

    # Ranking emojis: 🥇🥈🥉 then 🏅 for rest
    RANK_EMOJIS = ["🥇", "🥈", "🥉"]

    sorted_stocks = sorted(qualifying_stocks, key=lambda x: x['Score'], reverse=True)

    # Determine the condition threshold shown in header from top stock
    top_conditions = sorted_stocks[0]['Conditions'] if sorted_stocks else "?/16"

    now = datetime.now()
    header = (
        f"📊 *UPSTOX ELITE SWING SCANNER*\n"
        f"🕒 {now.strftime('%d %b %Y | %H:%M')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 *TOP PICKS ({top_conditions} Conditions)*\n"
    )

    blocks = []
    for i, stock in enumerate(sorted_stocks):
        rank_emoji = RANK_EMOJIS[i] if i < len(RANK_EMOJIS) else "🏅"

        symbol   = stock['Symbol']
        cmp_     = stock['CMP (₹)']
        target   = stock['Target 1 (₹)']
        sl       = stock['Stoploss (₹)']
        cond_str = stock['Conditions']          # e.g. "14/16"

        # Score percentage
        try:
            met, total = cond_str.split('/')
            score_pct = (int(met) / int(total)) * 100
        except Exception:
            score_pct = 0.0

        # Upside / downside %
        upside   = ((target - cmp_) / cmp_) * 100 if cmp_ else 0
        downside = ((cmp_ - sl) / cmp_) * 100 if cmp_ else 0

        # Risk : Reward  (reward / risk, rounded to 1 dp)
        risk   = cmp_ - sl
        reward = target - cmp_
        rr     = round(reward / risk, 1) if risk > 0 else 0.0

        block = (
            f"{rank_emoji} *{symbol}*\n"
            f"✅ Score: {score_pct:.1f}%\n"
            f"💰 CMP : ₹{cmp_:.2f}\n"
            f"🎯 Target : ₹{target:.2f} (+{upside:.2f}%)\n"
            f"🛑 Stoploss : ₹{sl:.2f} (-{downside:.2f}%)\n"
            f"📈 R:R : {rr} : 1"
        )
        blocks.append(block)

    separator = "\n\n" + "─" * 22 + "\n\n"
    return header + "\n" + separator.join(blocks) + "\n\n━━━━━━━━━━━━━━━━━━━━━━"

def format_whatsapp_message(qualifying_stocks):
    """
    Format results for WhatsApp via Green API — Premium Elite Swing Scanner layout.
    WhatsApp bold: *text*
    Ranking: 🥇🥈🥉 for top 3, 🏅 for rest.
    """
    RANK_EMOJIS = ["🥇", "🥈", "🥉"]

    if not qualifying_stocks:
        return "📊 *UPSTOX ELITE SWING SCANNER*\n\n❌ No stocks found matching criteria"

    sorted_stocks = sorted(qualifying_stocks, key=lambda x: x['Score'], reverse=True)

    top_cond = sorted_stocks[0]['Conditions']
    now      = datetime.now()

    lines = [
        "📊 *UPSTOX ELITE SWING SCANNER*",
        f"🕒 {now.strftime('%d %b %Y | %H:%M')}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"🏆 *TOP PICKS ({top_cond} Conditions)*",
        "",
    ]

    for i, stock in enumerate(sorted_stocks):
        rank   = RANK_EMOJIS[i] if i < len(RANK_EMOJIS) else "🏅"
        symbol = stock['Symbol']
        cmp_   = stock['CMP (₹)']
        target = stock['Target 1 (₹)']
        sl     = stock['Stoploss (₹)']
        cond   = stock['Conditions']

        try:
            met, total = cond.split('/')
            score_pct  = (int(met) / int(total)) * 100
        except Exception:
            score_pct  = 0.0

        upside   = ((target - cmp_) / cmp_) * 100 if cmp_ else 0
        downside = ((cmp_ - sl)    / cmp_) * 100 if cmp_ else 0

        risk   = cmp_ - sl
        reward = target - cmp_
        rr     = round(reward / risk, 1) if risk > 0 else 0.0

        lines += [
            f"{rank} *{symbol}*",
            f"✅ Score: {score_pct:.1f}%",
            f"💰 CMP : ₹{cmp_:.2f}",
            f"🎯 Target : ₹{target:.2f} (+{upside:.2f}%)",
            f"🛑 Stoploss : ₹{sl:.2f} (-{downside:.2f}%)",
            f"📈 R:R : {rr} : 1",
            "",
        ]

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)

def load_hardcoded_stocks():
    """Load hardcoded stock data"""
    if USE_POLARS:
        try:
            df = pl.read_csv(StringIO(DEFAULT_STOCKS))
            df = df.drop_nulls().to_pandas()
            return df
        except:
            pass
    return pd.read_csv(StringIO(DEFAULT_STOCKS), dtype=str).dropna().reset_index(drop=True)

def validate_csv_format(df):
    """Validate uploaded CSV format"""
    required_columns = ['instrument_key', 'tradingsymbol']
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for empty values
    if df['instrument_key'].isna().any() or df['tradingsymbol'].isna().any():
        return False, "Found empty values in required columns"
    
    # Check instrument_key format (basic validation)
    invalid_keys = df[~df['instrument_key'].str.contains('NSE_EQ|', na=False)]
    if len(invalid_keys) > 0:
        return False, f"Invalid instrument_key format found. Expected format: NSE_EQ|INE123456789"
    
    return True, "CSV format is valid"

def load_stocks_from_source(use_uploaded_csv, uploaded_file):
    """Load stocks from either uploaded CSV or hardcoded data"""
    if use_uploaded_csv and uploaded_file is not None:
        try:
            # Read uploaded CSV
            df = pd.read_csv(uploaded_file, dtype=str)
            df = df.dropna().reset_index(drop=True)
            
            # Validate format
            is_valid, message = validate_csv_format(df)
            if not is_valid:
                st.error(f"❌ CSV Validation Error: {message}")
                return None, None
            
            st.success(f"✅ Successfully loaded {len(df)} stocks from uploaded CSV")
            return df, "uploaded"
            
        except Exception as e:
            st.error(f"❌ Error reading uploaded CSV: {str(e)}")
            return None, None
    else:
        # Use hardcoded stocks
        df = load_hardcoded_stocks()
        return df, "hardcoded"

def main():
    st.markdown('<h1 class="main-header">🚀 Upstox Technical Scanner Ultra</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚡ Configuration")
    access_token = st.sidebar.text_input("Upstox Access Token", type="password")
    
    # CSV Upload Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Stock Data Source")
    
    use_uploaded_csv = st.sidebar.radio(
        "Choose stock data source:",
        ["Use Hardcoded Stocks", "Upload Custom CSV"],
        index=0
    ) == "Upload Custom CSV"
    
    uploaded_file = None
    if use_uploaded_csv:
        st.sidebar.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should contain 'instrument_key' and 'tradingsymbol' columns"
        )
        
        # Show CSV format requirements
        with st.sidebar.expander("📋 CSV Format Requirements"):
            st.markdown("""
            **Required Columns:**
            - `instrument_key`: NSE instrument key (e.g., NSE_EQ|INE585B01010)
            - `tradingsymbol`: Stock symbol (e.g., MARUTI)
            
            **Example CSV:**
            ```
            instrument_key,tradingsymbol
            NSE_EQ|INE585B01010,MARUTI
            NSE_EQ|INE139A01034,NATIONALUM
            ```
            """)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Load stocks based on selection
    df_instruments, data_source = load_stocks_from_source(use_uploaded_csv, uploaded_file)
    
    if df_instruments is None:
        st.error("❌ No stock data available. Please check your data source.")
        return
    
    # Show loaded data info
    if data_source == "uploaded":
        st.sidebar.success(f"✅ Using uploaded CSV ({len(df_instruments)} stocks)")
    else:
        st.sidebar.info(f"ℹ️ Using hardcoded stocks ({len(df_instruments)} stocks)")
    
    # Show sample data
    with st.sidebar.expander("👀 Preview Stock Data"):
        st.dataframe(df_instruments.head(10), use_container_width=True)
    
    # Configuration parameters
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_conditions = st.slider("Min Conditions", 1, 16, 10)
        max_workers = st.slider("Max Workers", 4, 32, 16)
    with col2:
        min_price = st.number_input("Min Price (₹)", value=50.0, min_value=1.0)
        min_volume = st.number_input("Min Volume", value=50000, min_value=1000)
    
    batch_size = st.sidebar.slider("Batch Size", 100, 1000, 500)
    send_whatsapp = st.sidebar.checkbox("📱 Send to WhatsApp", value=True)
    use_weighted_scoring = st.sidebar.checkbox("🎯 Weighted Scoring", value=False)
    
    # Main scan button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_analysis = st.button("🚀 Start Ultra Scan", type="primary", use_container_width=True)
    
    if start_analysis:
        if not access_token:
            st.error("❌ Please provide access token")
            return
        
        start_time = time.time()
        
        # Prepare instrument data
        try:
            instrument_keys = df_instruments["instrument_key"].str.strip().tolist()
            symbol_map = df_instruments.set_index("instrument_key")["tradingsymbol"].to_dict()
            st.success(f"✅ Loaded {len(instrument_keys)} instruments from {data_source} source")
        except Exception as e:
            st.error(f"❌ Error preparing instrument data: {e}")
            return
        
        # Fetch quotes asynchronously
        st.subheader("⚡ Fetching Market Data")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run async function
        quotes = asyncio.run(get_all_quotes_async(instrument_keys, access_token, batch_size))
        
        progress_bar.progress(1.0)
        status_text.text(f"Fetched {len(quotes)} quotes")
        
        if not quotes:
            st.error("❌ No quotes fetched. Check your access token.")
            return
        
        # Filter stocks using polars if available
        csv_data = []
        for key, quote in quotes.items():
            symbol = symbol_map.get(key, quote.get('symbol', ''))
            price = quote.get('last_price', 0)
            volume = quote.get('volume', 0)
            if price >= min_price and volume > min_volume:
                csv_data.append((symbol, price, volume))
        
        if USE_POLARS:
            df_filtered = pl.DataFrame(csv_data, schema=['symbol', 'price', 'volume']).sort('volume', descending=True).to_pandas()
        else:
            df_filtered = pd.DataFrame(csv_data, columns=['symbol', 'price', 'volume']).sort_values('volume', ascending=False)
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Stocks", len(quotes))
        with col2: st.metric("After Filters", len(df_filtered))
        with col3: st.metric("Filter Rate", f"{len(df_filtered)/len(quotes)*100:.1f}%" if len(quotes) > 0 else "0%")
        
        if len(df_filtered) == 0:
            st.error("❌ No stocks passed pre-filters")
            return
        
        # Technical analysis with weighted scoring
        st.subheader("🔥 Technical Analysis")
        qualifying_stocks = []
        
        # Setup weights (16 conditions now)
        if use_weighted_scoring:
            weights = np.array([1.2, 1.2, 1.0, 1.0, 1.1, 1.0, 0.8, 1.3, 1.4, 1.1, 0.9, 1.3, 0.8, 1.2, 1.0, 1.0])
        else:
            weights = np.ones(16)
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        args_list = [(row['symbol'], row['price'], row['volume'], min_conditions, weights) 
                     for _, row in df_filtered.iterrows()]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_stock_optimized, args): args 
                      for args in args_list}
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    qualifying_stocks.append(result)
                completed += 1
                progress_bar.progress(completed / len(args_list))
                status_text.text(f"Processed: {completed}/{len(args_list)} | Found: {len(qualifying_stocks)}")
        
        # Results
        end_time = time.time()
        total_time = end_time - start_time
        processing_rate = len(df_filtered) / total_time
        
        st.header("🎯 Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Qualifying Stocks", len(qualifying_stocks))
        with col2: st.metric("Success Rate", f"{len(qualifying_stocks)/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%")
        with col3: st.metric("Total Time", f"{total_time:.1f}s")
        with col4: st.metric("Processing Rate", f"{processing_rate:.1f}/sec")
        
        if not qualifying_stocks:
            st.warning("❌ No stocks qualified with current criteria")
            if send_whatsapp:
                wa_results = asyncio.run(send_whatsapp_message("*Scanner Results*\n\n❌ No stocks found matching criteria"))
                for ok, info in wa_results:
                    if ok:
                        st.info(f"📱 WhatsApp sent → {info}")
                    else:
                        st.warning(f"⚠️ WhatsApp failed → {info}")
            return
        
        # Sort and display results
        qualifying_stocks.sort(key=lambda x: x['Score'], reverse=True)
        results_df = pd.DataFrame(qualifying_stocks)
        
        st.subheader("🏆 Qualifying Stocks")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Send to WhatsApp
        if send_whatsapp:
            whatsapp_msg = format_whatsapp_message(qualifying_stocks)
            wa_results = asyncio.run(send_whatsapp_message(whatsapp_msg))
            for ok, info in wa_results:
                if ok:
                    st.success(f"📱 WhatsApp sent → {info}")
                else:
                    st.warning(f"⚠️ WhatsApp failed → {info}")
        
        # Download options
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download CSV", csv_data, 
                              f"upstox_scan_{timestamp}.csv", "text/csv", 
                              use_container_width=True)
        with col2:
            json_data = results_df.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button("📥 Download JSON", json_data,
                              f"upstox_scan_{timestamp}.json", "application/json",
                              use_container_width=True)

if __name__ == "__main__":
    main()
