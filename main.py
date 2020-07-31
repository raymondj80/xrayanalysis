import googleDriveFileLoader
import dataAnalyzer
import dataFormatter
import matplotlib.pyplot as plt

# Load Data 
filename1 = 'J465p6_T01_NdNiO3_STO_2hrs250C_15piecesCaH2.xrdml'
foldername1 = 'J465_NdNiO3_STO (p5 and p6)'
filename2 = 'SrTiO3_001_substrate.xrdml'
foldername2 = 'SrTiO3_001_substrate'

tth_thin_film = googleDriveFileLoader.fileLoader(filename1,foldername1)
tth_substrate = googleDriveFileLoader.fileLoader(filename2,foldername2)

Formatter = dataFormatter.Formatter()
Data = dataAnalyzer.Analyzer(tth_thin_film.createDict(),tth_substrate.createDict(),tth_thin_film.getKalpha2())
RAW_DATA = Data.FILM


# Analyze Data
Formatter.plotSemilogy(list(RAW_DATA.keys()),list(RAW_DATA.values()),20,60)
plt.show()
X_MIN = float(input("Enter Minimum X: "))
X_MAX = float(input("Enter Maximum X: "))
tth0 = Data.initializeTheta(Data.FILM,1,10)
tth_fit = Data.regressionFit(tth0,5e-5,0.01,X_MIN,X_MAX)
# dist = Data.braggsLaw(1,tth_fit[0],Data.KALPHA2)

# Plot Data
d = Formatter.jupyterFormatter(Data.FILM,Data.FIT)
FIT_DATA = Data.FIT
Formatter.plotSemilogy(list(RAW_DATA.keys()),list(RAW_DATA.values()),X_MIN,X_MAX)
Formatter.plotSemilogy(list(FIT_DATA.keys()),list(FIT_DATA.values()),X_MIN,X_MAX)
plt.show()
Formatter.createJupyterNB('test','Jupyter Notebooks', d)

googleDriveFileLoader.fileLoader.uploadFile('test.ipynb','Jupyter Notebooks/test.ipynb','application/x-ipynb+json')
