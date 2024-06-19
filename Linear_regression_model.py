import numpy as np
import pandas as pd
import os 
import plot as pl
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import gc
import matplotlib.pyplot as plt

class Preprocessing:
    def __init__(self):
        self.train_path =os.path.join('./Student dataset/','Student_Performance.csv')
        self.train_statictics_path ='./Statistics/Student_train/'
        self.test_statictics_path ='./Statistics/Student_results/'
        self.scaler=StandardScaler()
        self.normalizer=MinMaxScaler()

    def open_datasets(self,verbose=False):
        train_df = pd.read_csv(self.train_path)

        if verbose == True:
            print(f'\ntrain_df:\n\n{train_df.head(10)}')

        return train_df 
    
    def datasets_info(self,train_df,info=False,nulls=False,statistics=False):
        if info == True:
            print(f'\n{"-"*120}\n')
            print(f'\ntrain_df info:\n')
            print(f'\n\n{train_df.info()}\n')  
            print(f'train_df description:\n\n{train_df.describe()}\n')
            print(f'\n{"-"*120}\n')

        if nulls == True:
            print(f'\n{"-"*120}\n')
            print(f'train_df null values:\n\n{train_df.isnull().sum()}\n')
            print(f'\n{"-"*120}\n')

        if statistics == True:
            save = True
            show = True
            text = 'after'
            train_df_cm = train_df.corr()
            #Correlation matrix
            pl.plot_corr_matrix(train_df_cm,self.train_statictics_path,
                                f'Train data correlation matrix {text}',save=save,show=show)
            #Correlation matrix scatter plot
            pl.plot_corr_scatter(train_df,self.train_statictics_path,
                                f'Train data correlation scatter plot {text}',save=save,show=show)
            
    def save_predictions_to_csv(self,y_true, y_pred,Error, folder_dir, file_name):
        results_df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred,
            'Error' : Error
        })

        # Crear el directorio si no existe
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

        # Guardar el DataFrame en un archivo CSV
        file_path = os.path.join(folder_dir, f'{file_name}.csv')
        results_df.to_csv(file_path, index=False)

        print(f'Results saved to {file_path}')

    def drop_column(self,df,column_names=str,verbose=True):
        if isinstance(column_names, str):
            column_names = [column_names]

        df.drop(column_names,axis=1 , inplace=True)
        
        if verbose == True:
            print(f"Column : {column_names} was removed from the dataset")
        
        return df
    
    def set_col_index(self,df,column_name=str,verbose=True):
        # self.df=self.df.set_index([column_name] , inplace=True)
        df.set_index([column_name] , inplace=True)

        if verbose == True:
            print(f"Column : {column_name} was set as the index of the dataset")

        # print(f'\n{self.df.head(10)}\n') 
        return df
    
    def separate_data(self,df,text,verbose=False):
        if df is None:
            return print('No dataset availiable for split')

        else:
            idx = int(len(df) * 0.05)
            df = df[:idx]

            x_df = df.iloc[:,:-1]
           
            y_df = df.iloc[:,-1]

            x = x_df.to_numpy()
            y = y_df.to_numpy()

        if len(x) != len (y):
            return print('"x"  has not the same number of rows as "y"')

        if verbose == True:
            print(f'\n{"-"* 100}\n')
            print(f'\n"x" {text} dataset type and shape : {x.dtype} , {x.shape}  len {len(x)}\ncols = {x_df.columns.tolist()}:\n\n{x}\n') 
            print(f'\n"y" {text} dataset type and shape : {y.dtype} , {y.shape}  len {len(y)}\ncols = {y_df.name}:\n\n{y}\n')

        return x , y

    def run_preprocessing(self):
        train_df = self.open_datasets(verbose=True)

        train_df = train_df[['Previous Scores','Performance Index']]
        self.datasets_info(train_df,info=False,nulls=False,statistics=False)

        x , y = self.separate_data(train_df,'train',verbose=True)
        
        return x , y
    
class LinearRegressionModel:
    def __init__(self, X, y, learning_rate=0.01, momentum=0.9, epochs=1000, tolerance=1e-6):
        self.lr = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.tolerance = tolerance
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.epsilon = 1e-8

    def MSE(self, y, y_pre, n):
        return np.sum((y - y_pre) ** 2) / n

    def gradients(self, X, y, n_samples, weights, bias):
        predictions = np.dot(X, weights) + bias
        errors = y - predictions

        Wg = -(2/n_samples) * np.dot(X.T, errors)
        Bg = -(2/n_samples) * np.sum(errors)

        return Wg, Bg

    def GDM(self, Wg_new, Bg_new, Lr, Momentum, velocity_W, velocity_bias):
        clip_value = 1.0
        Wg_new = np.clip(Wg_new, -clip_value, clip_value)
        Bg_new = np.clip(Bg_new, -clip_value, clip_value)

        velocity_W = Momentum * velocity_W + Lr * Wg_new
        velocity_bias = Momentum * velocity_bias + Lr * Bg_new

        Wg_new -= velocity_W
        Bg_new -= velocity_bias

        return Wg_new, Bg_new, velocity_W, velocity_bias

    def fit_GDM(self, verbose=False):
        weights = np.zeros(self.n_features)
        bias = 0
        velocity_W = np.zeros(self.n_features)
        velocity_bias = 0
        Errors = []

        for epoch in range(self.epochs):
            Wg, Bg = self.gradients(self.X, self.y, self.n_samples, weights, bias)
            weights, bias, velocity_W, velocity_bias = self.GDM(Wg, Bg, self.lr, self.momentum, velocity_W, velocity_bias)

            if np.linalg.norm(Wg) < self.tolerance and abs(Bg) < self.tolerance:
                return weights, bias
            
            if verbose:
                y_pre = self.predict(self.X, weights, bias)
                mse = self.MSE(self.y, y_pre, self.n_samples)
                Errors.append(mse)
                print(f'\n{"-"* 100}\n')
                print(f'\nEpoch : {epoch}\nWeights coefficients :\n{weights}\n\nBias coefficients :\n{bias}\n\nMSE = {mse}\n')


        return weights, bias , Errors
    
    def fit_ADAM(self,verbose=False):
        weights = np.zeros(self.n_features)
        bias = 0
        mw = np.zeros(self.n_features)  
        vw = np.zeros(self.n_features) 
        mb = 0
        vb = 0 
        Beta1 = 0.9 # Exponential decay for the first momentum 
        Beta2 = 0.999 # Exponential decay for the second momentum 
        tolerance = self.tolerance
        Errors = []

        for epoch in range(self.epochs):
            Wg, Bg = self.gradients(self.X, self.y, self.n_samples, weights, bias)

            #Update momentums
            mw_new = (Beta1 * mw) + ((1 - Beta1) * Wg)
            vw_new = (Beta2 * vw) + ((1 - Beta2) * Wg**2)

            mb_new = (Beta1 * mb) + ((1 - Beta1) * Bg)
            vb_new = (Beta2 * vb) + ((1 - Beta2) * Bg**2)

            #Bias correction
            Mw_corrected = mw_new / (1 - Beta1**(epoch + 1))
            Vw_corrected = vw_new / (1 - Beta2**(epoch + 1))

            Mb_corrected = mb_new / (1 - Beta1**(epoch + 1))
            Vb_corrected = vb_new / (1 - Beta2**(epoch + 1))

            #update parameters
            weights = weights - self.lr * (Mw_corrected / (np.sqrt(Vw_corrected) + self.epsilon))

            bias = bias - self.lr * (Mb_corrected / (np.sqrt(Vb_corrected) + self.epsilon))

            if np.linalg.norm(Wg) < tolerance and abs(Bg) < tolerance:
                print(f'Early stopping at epoch {epoch}')
                return weights , bias

            if verbose == True:
                y_pred = self.predict(self.X, weights, bias)
                mse = self.MSE(self.y, y_pred, self.n_samples)
                Errors.append(mse)
                print(f'Epoch {epoch + 1}\nMSE: {mse:.4f}')

        return weights , bias , Errors

    def predict(self, X, W, b):
        y_pred = np.round(np.abs(np.dot(X, W) + b),2)
        # print(f'Prediction = {y_pred}')
        return y_pred

if __name__ == '__main__':
    # Preprocessing
    pre = Preprocessing()
    x, y = pre.run_preprocessing()

    model = LinearRegressionModel(x, y, learning_rate=0.001, momentum=0.9, epochs=1000, tolerance=1e-6)
    save = False
    plot = True
    GD_momentum = True
    Adam = True 
    plot_MSE = True
    
    if GD_momentum == True:
        #With momentum
        weights, bias , Errors_GDM = model.fit_GDM(verbose=True)
        y_pred = model.predict(x,weights,bias)
        # print(f'\nPrediction = {y_pred}')

        Error = np.round(np.abs(y - y_pred), 2)
        # print(f'\nError = {Error}')

        if save == True:
            pre.save_predictions_to_csv(y,y_pred,Error,'./Student dataset/results/','Linear regression with GDM')

        if plot == True:
            pl.plot_slope(x, y, y_pred, folder_dir='./Statistics/Student_results/', file_name='Linear regression with GDM scatterplot', save=False, show=True)


    if Adam == True:
        #ADAM
        weights, bias , Errors_ADAM = model.fit_ADAM(verbose=True)
        y_pred = model.predict(x,weights,bias)
        # print(f'\nPrediction = {y_pred}')

        Error = np.round(np.abs(y - y_pred), 2)
        # print(f'\nError = {Error}')

        if save == True:
            pre.save_predictions_to_csv(y,y_pred,Error,'./Student dataset/results/','Linear regression with ADAM')

        if plot == True:
            pl.plot_slope(x, y, y_pred, folder_dir='./Statistics/Student_results/', file_name='Linear regression with ADAM scatterplot', save=False, show=True)
            

    if plot_MSE == True:
            pl.plot_error2(Errors_GDM,Errors_ADAM, folder_dir='./Statistics/Student_results/', file_name='MSE error with GDM vs ADAM', save=False, show=True)
