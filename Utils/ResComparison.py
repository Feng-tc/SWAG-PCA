import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

disk = 'L:'

path = disk + '/res/TestSave/'

lst_name = os.listdir(path)

lst_model_name = []
lst_BE = []
lst_Jac = []
lst_Dice = []
lst_train_info_mean = []
lst_train_info_best = []

for i in range(len(lst_name)):
    df = pd.read_excel(path + lst_name[i], sheet_name='ed_to_es')
    model_name = df.iloc[0, 1]
    lst_model_name.append(model_name.split('_')[2])

    dataset_name = model_name.split('_')[-2]
    if dataset_name == 'AYM':
        if len(lst_Dice) == 0:
            lst_BE.append(['ACDC', 'York', 'MICCAI', 'ALL'])
            lst_Jac.append(['ACDC', 'York', 'MICCAI', 'ALL'])
            lst_Dice.append(['ACDC', 'York', 'MICCAI', 'ALL'])
        lst_BE.append(df.iloc[3 : 7, 1].tolist())
        lst_Jac.append(df.iloc[3 : 7, 2].tolist())
        lst_Dice.append(df.iloc[3 : 7, 3].tolist())
    else:
        pass
    
    df = pd.read_excel(path + lst_name[i], header=None, sheet_name='training')
    if len(lst_train_info_best) == 0:
        lst_train_info_best.append(['loss',	'similarity_loss', 'lcc', 'kl_loss', 'kl', 'nu_kl', 'u_kl', 'jacobian_loss', 'L2', 'Dice', 'Epoch'])
    lst_train_info_mean.append(df.iloc[0, : 10].tolist())
    lst_train_info_best.append(df.iloc[1, : 11].tolist())

lst_BE = np.array(lst_BE).T.tolist()
lst_Jac = np.array(lst_Jac).T.tolist()
lst_Dice = np.array(lst_Dice).T.tolist()

lst_train_info_mean = np.array(lst_train_info_mean).T.tolist()
lst_train_info_best = np.array(lst_train_info_best).T.tolist()

columns = ['Dataset'] + lst_model_name

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))

table1 = ax1.table(cellText=lst_BE, colLabels=columns, loc='upper left', cellLoc='center')
ax1.axis('off')
ax1.set_title('BE')
table1.auto_set_font_size(False)
table1.set_fontsize(10)

table2 = ax2.table(cellText=lst_Jac, colLabels=columns, loc='upper left', cellLoc='center')
ax2.axis('off')
ax2.set_title('Jac')
table2.auto_set_font_size(False)
table2.set_fontsize(10)

table3 = ax3.table(cellText=lst_Dice, colLabels=columns, loc='upper left', cellLoc='center')
ax3.axis('off')
ax3.set_title('Dice')
table3.auto_set_font_size(False)
table3.set_fontsize(10)

table4 = ax4.table(cellText=lst_train_info_best, colLabels=['Metric'] + lst_model_name, loc='upper left', cellLoc='center')
ax4.axis('off')
ax4.set_title('Train Info (Best)')
table4.auto_set_font_size(False)
table4.set_fontsize(10)

plt.tight_layout()
plt.show()


