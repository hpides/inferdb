import sys
import os
from pathlib import Path 
project_folder = Path(__file__).resolve().parents[3]
sys.path.append(str(project_folder))
src_folder = os.path.join(project_folder, 'src')
sys.path.append(str(src_folder))
from encoder import Encoder
from optimizer import Problem
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from copy import deepcopy

exp_folder = Path(__file__).resolve().parents[3]
data_path = os.path.join(exp_folder, 'data', 'paper_data', 'mnist_784', 'mnist_784.csv')

df = pd.read_csv(data_path)

training_features = [i for i in list(df) if i != 'class']

plt.rcParams.update({'text.usetex' : True
                    , 'pgf.rcfonts': False
                    , 'text.latex.preamble':r"""\usepackage{iftex}
                                            \ifxetex
                                                \usepackage[libertine]{newtxmath}
                                                \usepackage[tt=false]{libertine}
                                                \setmonofont[StylisticSet=3]{inconsolata}
                                            \else
                                                \RequirePackage[tt=false, type1=true]{libertine}
                                            \fi"""   
                    })

##################### Binary analysis. Idea: select features for each class as a binary problem. Then check the histogram of the features
##################### i.e., how many time a feature is shared across classes
#####################

num_row = 2
num_col = 5

# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(10):

    model = xgb.XGBClassifier(objective="binary:logistic", n_jobs=-1, random_state=42)

    df_copy = deepcopy(df)
    df_copy['class'] = df_copy.apply(lambda x: 1 if i == x['class'] else 0, axis=1)

    X = df_copy[training_features]
    y = df_copy['class'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)

    encoder = Encoder('classification')

    encoder.fit(X_train.to_numpy(), y_pred_train, [])

    X_train_encoded = encoder.transform_dataset(X_train, [i for i in range(X_train.shape[1])])

    problem = Problem(X_train_encoded, y_pred_train, encoder.num_bins, 'classification', 1)

    problem.set_costs()

    class_representative = problem.iv_array.reshape((28,28))
    

    ax = axes[i//num_col, i%num_col]
    ax.imshow(class_representative, cmap='gray')
    ax.set_title('Label: {}'.format(i))

plt.tight_layout()
plt.show()




