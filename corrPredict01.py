#corrPredict01.py
import numpy as np
import scipy as sc
import math
import pandas as pd
#6/7/18
#load StockCorrelation
#Assuming for future use that StockCorrelation will be a pandas dff
L = StockCorrelations.shape[0]

def low(x,y):
	return x*y - math.sqrt((1-x**2)*(1-y**2))

def high(x,y):
	return x*y + math.sqrt((1-x**2)*(1-y**2))

Dates = np.array(StockCorrelations['Dates'])
D1    = strsplit(Dates[1][1], '_to_')
Start = D1{1}
D2    = strsplit(Dates[120][1], '_to_')
Stop = D2{2}

% Make Directory
DIR = '/Users/pauldavid/Documents/My_Files/School_Work_and_Research/CGU/Thesis/Thesis_CodesAlgorithms/Thesis_Matlab/All_Drafts/SVM_Stocks/CorrPredict01';
if exist(DIR, 'dir') == 0
    mkdir(DIR)
    addpath(DIR)
end

for i in range(L-1):#= 1:(L-1)
    for j in range(i+1, L):#= (i+1):L
        # Find x and y 'coordinates' for suppose 3x3 correlation matrix
        X = reshape(StockCorrelations{i}.Cor(1,2,:), [120 1])
        Y = reshape(StockCorrelations{j}.Cor(1,2,:), [120 1])
        
        Tick1 = strsplit(StockCorrelations{i}.Ticker, '_vs_')
        Tick2 = strsplit(StockCorrelations{j}.Ticker, '_vs_')
        newTick = strcat(Tick1{2}, '_vs_', Tick2{2}) # % New ticker to compare
        
        # Construct actual correlations
        Open1       = StockCorrelations{i}.StockOpen
        Open2       = StockCorrelations{j}.StockOpen
        EmpCorr     = zeros(2,2,120)
        PredictCorr = zeros(2,2,120)
        for k in range(120):#= 1:120
            EmpCorr(:,:,k)     = corrcoef(Open1(k:k+4), Open2(k:k+4))
            Sample             = (High(X(k),Y(k)) - Low(X(k),Y(k)) )*rand(1) + Low(X(k),Y(k))
            PredictCorr(:,:,k) = [1, Sample;Sample,1]
        #end
        
        # Make subdirectories, plot images.
        subDir = strcat(DIR, '/', newTick)
        if exist(subDir, 'dir') == 0:
            mkdir(subDir)
            addpath(subDir)
        #end
        
        M2017_02_23_EllipseGrid(EmpCorr, [Tick1{2}, '/', Tick2{2}, ' Correlations: ', Start, ' to ', Stop])
        saveas(gcf, strcat(subDir, '/', newTick, '_EmpiricalCorr'), 'fig')
        saveas(gcf, strcat(subDir, '/', newTick, '_EmpiricalCorr'), 'epsc')
        
        M2017_02_23_EllipseGrid(PredictCorr, [Tick1{2}, '/', Tick2{2}, ' Predicted Correlations: ', Start, ' to ', Stop])
        saveas(gcf, strcat(subDir, '/', newTick, '_PredictCorr'), 'fig')
        saveas(gcf, strcat(subDir, '/', newTick, '_PredictCorr'), 'epsc')
        
        # Construct Data Cell
        Data = {}
        Data['Ticker']         = newTick
        Data['Dates']          = Dates
        Data['AAPL_vs_Stock1'] = X
        Data['AAPL_vs_Stock2'] = Y
        Data['Stock1Open']     = Open1
        Data['Stock2Open']     = Open2
        Data['EmpCorr']        = EmpCorr
        Data['PredictCorr']    = PredictCorr
        
        CorrPredict{i,j} = Data;
        
        close all
    end
end

save('CorrPredict.mat', 'CorrPredict')




































