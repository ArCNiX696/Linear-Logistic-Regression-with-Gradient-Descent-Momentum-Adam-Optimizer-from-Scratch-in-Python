import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_corr_matrix(corr_matrix,folder_dir,file_name,save=False,show=True):
    plt.figure(figsize=(10,10))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.4f') #annot=show corr values ,cmap= colors, fmt=decimals 
    plt.title('Correlation Matrix')
    if save == True:
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

        plot_path = os.path.join(folder_dir,f'{file_name}.png')
        plt.savefig(plot_path)

    if show == True:
        plt.tight_layout()
        plt.show()

    plt.close()
    gc.collect()


def boxplot(df,target_variable,folder_dir,file_name,save=False,show=True):
    # Create a figure and axes for the plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Create the boxplot for each column with respect to the target variable
    sns.boxplot(data=df.drop(columns=target_variable), ax=ax)
    sns.swarmplot(data=df.drop(columns=target_variable), color=".25", ax=ax)
    
    # Set the title and axes labels
    ax.set_title('Boxplot of distribution with respect to ' + target_variable)
    ax.set_ylabel('Distributon')
    
    # Rotate x-axis labels for better visualization
    plt.xticks(rotation=45)

    if save == True:
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

        plot_path = os.path.join(folder_dir,f'{file_name}.png')
        plt.savefig(plot_path)
    
    if show == True:    
        # Show the plot
        plt.tight_layout()
        plt.show()

    plt.close()
    gc.collect()


def plot_corr_scatter(df,folder_dir,file_name,save=False,show=True):
    sns.set_theme(style='whitegrid')
    # plt.figure(figsize=(12,12))
    g=sns.pairplot(df,plot_kws={'s': 3},height=2.5)
    g.figure.suptitle('Linear Correlation between feaures', size=16)
    g.figure.subplots_adjust(top=0.93)
    
    if save == True:
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

        plot_path = os.path.join(folder_dir,f'{file_name}.png')
        plt.savefig(plot_path)

    if show == True:
        plt.tight_layout()
        plt.show()

    plt.close()
    gc.collect()

def plot_slope(X,y,y_pred,folder_dir,file_name,save=False,show=True):
    plt.scatter(X, y, color='blue',label='Data Points')
    plt.plot(X, y_pred, color='red')
    plt.title(f'Optimized slope by {file_name}')

    if save == True:
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

        plot_path = os.path.join(folder_dir,f'{file_name}.png')
        plt.savefig(plot_path)

    if show == True:
        plt.tight_layout()
        plt.show()

    plt.close()
    gc.collect()


def plot_error(errors,opt_name,folder_dir,file_name,save=False,show=True):
    epochs_1 = range(1, len(errors)+1)
    
    plt.figure(figsize=(10,10))
    plt.plot(epochs_1,errors,label=f'{opt_name}',color='red')
    plt.title(f'{opt_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    
    if save == True:
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

        plot_path = os.path.join(folder_dir,f'{file_name}.png')
        plt.savefig(plot_path)

    if show == True:
        plt.tight_layout()
        plt.show()

    plt.close()
    gc.collect()

def plot_error2(errors_1,errors_2,folder_dir,file_name,save=False,show=True):
    epochs_1 = range(1, len(errors_1)+1)
    epochs_2 = range(1, len(errors_2)+1)

    plt.figure(figsize=(10,10))
    plt.plot(epochs_1,errors_1,label='GDM',color='red')
    plt.plot(epochs_2,errors_2,label='ADAM',color='blue')
    plt.title("MSE")
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    
    if save == True:
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

        plot_path = os.path.join(folder_dir,f'{file_name}.png')
        plt.savefig(plot_path)

    if show == True:
        plt.tight_layout()
        plt.show()

    plt.close()
    gc.collect()

def plot_confusion_matrix(y_true, y_pred, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    # Ensure y_true and y_pred are 1D arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm[::-1, ::-1] #Reorder rows and cols
    # print(f'cm:\n{cm}')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a DataFrame from the confusion matrix for easier plotting
    labels = sorted(set(y_true) | set(y_pred))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
   
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=['Predicted Positive', 'Predicted Negative'], yticklabels=['Actual Positive', 'Actual Negative'])
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
