import googleDriveFileLoader
import dataAnalyzer

# Load Data 
filename1 = 'J465p6_T01_NdNiO3_STO_2hrs250C_15piecesCaH2.xrdml'
foldername1 = 'J465 CaH2 test'
filename2 = 'SrTiO3_001_substrate.xrdml'
foldername2 = 'SrTiO3_001_substrate'

tth_thin_film = googleDriveFileLoader.fileLoader(filename1,foldername1)
tth_substrate = googleDriveFileLoader.fileLoader(filename2,foldername2)

# Analyze Data
Data = dataAnalyzer.Analyzer(tth_thin_film.createDict(),tth_substrate.createDict(),tth_thin_film.getKalpha2()) 
subtracted_data = Data.pseudoVoigt(Data.FILM,0.05,2.7,1) - Data.pseudoVoigt(Data.SUBSTRATE,0.05,2.7,1)
tth_fit = Data.regressionFit(Data.initializeTheta(Data.FILM,1,10),subtracted_data,5e-5,0.01,23,27)
dist = Data.braggsLaw(1,tth_fit[0],Data.KALPHA2)

