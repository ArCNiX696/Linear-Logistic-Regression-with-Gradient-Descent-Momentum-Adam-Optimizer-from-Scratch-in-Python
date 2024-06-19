import numpy as np
import pandas as pd
import os 
import plot as pl
from sklearn.preprocessing import MinMaxScaler

normalizer = MinMaxScaler()

class Preprocessing:
    def __init__(self):
        #Train datasets 
        self.X_train_path =os.path.join('./Titanic data/','train_X.csv')
        self.y_train_path =os.path.join('./Titanic data/','train_Y.csv')
        #Test datasets
        self.X_test_path =os.path.join('./Titanic data/','test_X.csv')
        self.y_test_path =os.path.join('./Titanic data/','test_Y.csv')
   
    def open_datasets(self,see_train=False,see_test=False):
        #Train datasets
        X_train_df = pd.read_csv(self.X_train_path)
        y_train_df = pd.read_csv(self.y_train_path)

        if see_train == True:
            print(f'\n\ntrain_df:\n\n{X_train_df.head(10)}')
            print(f'\n\ntrain_df:\n\n{y_train_df.head(10)}')

        #Test datasets
        X_test_df = pd.read_csv(self.X_test_path)
        y_test_df = pd.read_csv(self.y_test_path)

        if see_test == True:
            print(f'\n\ntrain_df:\n\n{X_test_df.head(10)}')
            print(f'\n\ntrain_df:\n\n{y_test_df.head(10)}')

        return X_train_df , y_train_df , X_test_df , y_test_df
    
    def datasets_info(self,X_df,y_df,Text,info=False,nulls=False):
        if info == True:
            print(f'\n{"-"*120}\n')
            #X_train_df
            print(f'\nX_{Text}_df info:\n')
            print(f'\n\n{X_df.info()}\n')  
            print(f'X_{Text}_df description:\n\n{X_df.describe()}\n')
            #y_train_df
            print(f'\ny_{Text}_df info:\n')
            print(f'\n\n{y_df.info()}\n')  
            print(f'y_{Text}_df description:\n\n{y_df.describe()}\n')
            print(f'\n{"-"*120}\n')

        if nulls == True:
            print(f'\n{"-"*120}\n')
            print(f'X_{Text}_df null values:\n\n{X_df.isnull().sum()}\n')
            print(f'y_{Text}_df null values:\n\n{y_df.isnull().sum()}\n')
            print(f'\n{"-"*120}\n')

    def drop_column(self,df,column_names=str,verbose=True):
        if isinstance(column_names, str):
            column_names = [column_names]

        df.drop(column_names,axis=1 , inplace=True)
        
        if verbose == True:
            print(f"Column : {column_names} was removed from the dataset")
        
        return df
    
    def remove_rows_with_nulls(self,df: pd.DataFrame, verbose: bool=False) -> pd.DataFrame:
        # Count the number of null elements
        total_nulls = df.isnull().sum().sum()
        
        # Count the number of rows before removing rows with null values
        rows_before = df.shape[0]

        # Remove rows with null values in any column
        df_without_nulls = df.dropna()

        # Count the number of rows after removing rows with null values
        rows_after = df_without_nulls.shape[0]

        if verbose:
            print(f"Total null elements found: {total_nulls}")
            print(f"Total rows removed: {rows_before - rows_after}")

        return df_without_nulls
    
    def merge_and_clean_dataframes(self, df_independent: pd.DataFrame, df_dependent: pd.DataFrame, verbose: bool = False):
        # Merge the DataFrames using their indices
        merged_df = pd.merge(df_independent, df_dependent, left_index=True, right_index=True)

        if verbose:
            print(f"DataFrame shape before cleaning: {merged_df.shape}")

        # Remove rows with null values
        cleaned_df = self.remove_rows_with_nulls(merged_df, verbose)

        # Separate the cleaned DataFrame back into independent and dependent DataFrames
        cleaned_independent = cleaned_df[df_independent.columns]
        cleaned_dependent = cleaned_df[df_dependent.columns]

        if verbose:
            print(f"\ncleaned_independent shape:\n {cleaned_independent.shape}\n\ncleaned_dependent shape:\n {cleaned_dependent.shape}")

        return cleaned_independent, cleaned_dependent
    
    def set_col_index(self,df,column_name=str,verbose=True):
        # self.df=self.df.set_index([column_name] , inplace=True)
        df.set_index([column_name] , inplace=True)

        if verbose == True:
            print(f"Column : {column_name} was set as the index of the dataset")

        # print(f'\n{self.df.head(10)}\n') 
        return df
    
    def separate_data(self,X_train_df,y_train_df,text,verbose=False):
        if X_train_df is None:
            return print(f'Dataset "X_{text}_df" is not availiable for split')
        
        if y_train_df is None:
            return print(f'Dataset "y_{text}_df" is not availiable for split')
        else:
            x = X_train_df.to_numpy()
            y = y_train_df.to_numpy()

        if len(x) != len (y):
            return print('"x"  has not the same number of rows as "y"')

        if verbose == True:
            print(f'\n{"-"* 100}\n')
            print(f'\n"x" {text} dataset type and shape : {x.dtype} , {x.shape}  len {len(x)}\ncols = {X_train_df.columns.tolist()}:\n\n{x}\n') 
            print(f'\n"y" {text} dataset type and shape : {y.dtype} , {y.shape}  len {len(y)}\ncols = {y_train_df.columns.tolist()}:\n\n{y}\n')

        return x , y
    
    def run_preprocessing(self):
        #Import datasets
        X_train_df , y_train_df , X_test_df , y_test_df = self.open_datasets(see_train=False , see_test=True)

        #Set column Id as index
        X_train_df = self.set_col_index(X_train_df,'Id',verbose=False)
        y_train_df = self.set_col_index(y_train_df,'Id',verbose=False)
        X_test_df = self.set_col_index(X_test_df,'Id',verbose=False)
        y_test_df = self.set_col_index(y_test_df,'Id',verbose=False)

        #Delete rows with null elements
        X_test_df , y_test_df = self.merge_and_clean_dataframes(X_test_df , y_test_df,verbose=True)

        #Normalization
        columns_to_normalize = ['Age' , 'Fare']
        X_train_df[columns_to_normalize] = normalizer.fit_transform(X_train_df[columns_to_normalize])
        X_test_df[columns_to_normalize] = normalizer.fit_transform(X_test_df[columns_to_normalize])
       
        self.datasets_info(X_train_df,y_train_df,'train',info=False,nulls=False)
        self.datasets_info(X_test_df,y_test_df,'test',info=False,nulls=False)

        X_train , y_train = self.separate_data(X_train_df,y_train_df,'train',verbose=False)
        X_test , y_test = self.separate_data(X_test_df,y_test_df,'test',verbose=False)
    
        return X_train , y_train , X_test , y_test 
    
class LogisticRegression:
    def __init__(self, learning_rate=0.001, epochs=1000,tolerance=1e-6):
        self.lr = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.losses = []
        self.epsilon = 1e-8
         
    # Sigmoid method
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # binary cross entropy
    def BCE(self, y_true, y_pred):
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def feed_forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return y_pred

    def fit_ADAM(self, X, y,verbose=False): #y : (891, 1)
        n_samples, n_features = X.shape #X : (891, 7)
        self.weights = np.zeros(n_features) #shape:(7,)
        self.bias = 0
        mw = np.zeros(n_features)  
        vw = np.zeros(n_features) 
        mb = 0
        vb = 0 
        Beta1 = 0.9 # Exponential decay for the first momentum 
        Beta2 = 0.999 # Exponential decay for the second momentum 
        tolerance = self.tolerance
    
        for epoch in range(self.epochs):
            y_pred = self.feed_forward(X) #A shape : (891,)
            bce = self.BCE(y, y_pred)
            self.losses.append(bce)
            
            y = y.ravel()
            dz = (y_pred - y) # shape = (891, 891)
           
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, dz) #dw shape : (7, 891)
           
            db = (1 / n_samples) * np.sum(y_pred - y) #derivate of bias it only got 1 value
           
            dw_new = (Beta1 * mw) + ((1 - Beta1) * dw)
            vw_new = (Beta2 * vw) + ((1 - Beta2) * dw**2)

            db_new = (Beta1 * mb) + ((1 - Beta1) * db)
            vb_new = (Beta2 * vb) + ((1 - Beta2) * db**2)

            #Bias correction
            dw_corrected = dw_new / (1 - Beta1**(epoch + 1))
            vw_corrected = vw_new / (1 - Beta2**(epoch + 1))

            db_corrected = db_new / (1 - Beta1**(epoch + 1))
            vb_corrected = vb_new / (1 - Beta2**(epoch + 1))

            #update parameters
            self.weights = self.weights - self.lr * (dw_corrected / (np.sqrt(vw_corrected) + self.epsilon))

            self.bias = self.bias - self.lr * (db_corrected / (np.sqrt(vb_corrected) + self.epsilon))

            if np.linalg.norm(dw) < tolerance and abs(db) < tolerance:
                print(f'Early stopping at epoch {epoch}')
                return self.weights , self.bias
            
            if verbose == True:
                print(f'\n{"-"* 100}\n')
                print(f'Epoch {epoch + 1}\nBCE: {bce:.4f}')

        return self.weights , self.bias , self.losses

            
    def predict(self, X):
        threshold = 0.5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted]
        
        return np.array(y_predicted_cls)

if __name__ == "__main__":
    pre = Preprocessing()
    #HERE
    X_train , y_train , X_test , y_test = pre.run_preprocessing()

    train_statictics_path ='./Statistics/Titanic train results/'
    test_statictics_path ='./Statistics/Titanic test results/'

    fit = True
    if fit == True:
        #ADAM optimization for logistic regression model
        model = LogisticRegression(learning_rate=0.01, epochs=1000,tolerance=1e-6)
        _ ,_ , losses = model.fit_ADAM(X_train, y_train,verbose=True)
        
        plot = True
        save = False
        show = True
        predict = True
        confusion_matrix = True
        save_pred = False

        if plot == True:
            pl.plot_error(losses,'BCE error Optimized with ADAM',folder_dir=test_statictics_path, file_name='BCE error optimized by ADAM',save=save, show=show)
        
        if predict:
            y_pred = model.predict(X_test)
            print(f'\n{"-"* 100}\n')
            print(f"\nPredictions:\n{y_pred}\n")

            # Crear DataFrame para guardar resultados
            results = pd.DataFrame({
                'Actual': y_test.ravel(),
                'Predicted': y_pred
            })

            conditions = [
                (results['Actual'] == 1) & (results['Predicted'] == 1),
                (results['Actual'] == 0) & (results['Predicted'] == 0),
                (results['Actual'] == 1) & (results['Predicted'] == 0),
                (results['Actual'] == 0) & (results['Predicted'] == 1)
            ]
            choices = ['TP', 'TN', 'FN', 'FP']
            results['Classification'] = np.select(conditions, choices)

            accuracy = (results['Actual'] == results['Predicted']).mean()
            print(f"\nAccuracy: {accuracy:.2f}\n\n")
            print(f'\n{"-"* 100}\n')
            print(results['Classification'].value_counts())
            print(f'\n{"-"* 100}\n')
            print(results.head())
            print(f'\n{"-"* 100}\n')
        
            if save_pred:
                results.to_csv(os.path.join(test_statictics_path, 'predictions_with_classifications.csv'), index=False)
        

        if confusion_matrix == True:
            pl.plot_confusion_matrix(y_test,y_pred,normalize=True)

            

    

