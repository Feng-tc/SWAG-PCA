import matplotlib.pyplot as plt

import numpy as np

import torch

def plt_everything(array, name='test', title=None, scale=None, overlapped=False, multi_overlapped=False, cm=plt.cm.gray, show=None, display=False):
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    elif isinstance(array, list):
        array = [i.cpu().numpy() for i in array]

    plt.close()

    if multi_overlapped == True:
        if show == 'sc':
            # 单列显示
            c = 1
            r = len(array)
            plt.figure(figsize=[c * 2.5, r * 2])
            for i in range(len(array)):
                plt.subplot(r, c, i + 1)
                plt.imshow(array[i][0], cmap=plt.cm.gray)
                if scale == None:
                    plt.imshow(array[i][1], cmap=plt.cm.jet, alpha=0.5)
                else:
                    plt.imshow(array[i][1], cmap=plt.cm.jet, alpha=0.5, vmin=scale[0], vmax=scale[1])
                plt.colorbar()
                if title != None:
                    plt.title(title[i])
        elif show == 'sr':
            # 单行显示
            c = len(array)
            r = 1
            plt.figure(figsize=[c * 2.5, r * 2])
            for i in range(len(array)):
                plt.subplot(r, c, i + 1)
                plt.imshow(array[i][0], cmap=plt.cm.gray)
                if scale == None:
                    plt.imshow(array[i][1], cmap=plt.cm.jet, alpha=0.5)
                else:
                    plt.imshow(array[i][1], cmap=plt.cm.jet, alpha=0.5, vmin=scale[0], vmax=scale[1])
                plt.colorbar()
                if title != None:
                    plt.title(title[i])
        
        plt.tight_layout()
        if display == True:
            plt.show()
        else:
            plt.savefig('img/' + name + '.png')
        
        print('Done')
        return 0

    shapes = array.shape

    if shapes[0] == 1:
        plt.figure(figsize=[2.5, 2])
        if scale == None:
            plt.imshow(array[0], cmap=cm)
        else:
            plt.imshow(array[0], cmap=cm, vmin=scale[0], vmax=scale[1])
        plt.colorbar()
        if title != None:
            plt.title(title[0])
    elif shapes[0] == 2 and overlapped:
        plt.figure(figsize=[2.5, 2])
        plt.imshow(array[0], cmap=plt.cm.gray)
        plt.imshow(array[1], cmap=plt.cm.jet, alpha=0.5)
        if title != None:
            plt.title(title[0])
        plt.colorbar()
    elif shapes[0] > 1 and overlapped == False:
        if show == 'sc':
            # 单列显示
            c = 1
            r = shapes[0]
            plt.figure(figsize=[c * 2.5, r * 2])
            for i in range(0, shapes[0]):
                plt.subplot(r, c, i + 1)
                if scale == None:
                    plt.imshow(array[i], cmap=cm)
                else:
                    plt.imshow(array[i], cmap=cm, vmin=scale[0], vmax=scale[1])
                plt.colorbar()
                if title != None:
                    plt.title(title[i])
        elif show =='sr':
            # 单行显示
            c = shapes[0]
            r = 1
            plt.figure(figsize=[c * 2.5, r * 2])
            for i in range(0, shapes[0]):
                plt.subplot(r, c, i + 1)
                if scale == None:
                    plt.imshow(array[i], cmap=cm)
                else:
                    plt.imshow(array[i], cmap=cm, vmin=scale[0], vmax=scale[1])
                plt.colorbar()
                if title != None:
                    plt.title(title[i])
        else:
            c = 4
            r = shapes[0] // c + 1 if shapes[0] // c != 0 else 1
            plt.figure(figsize=[c * 2.5, r * 2])
            for i in range(0, shapes[0]):
                plt.subplot(r, c, i + 1)
                if scale == None:
                    plt.imshow(array[i], cmap=cm)
                else:
                    plt.imshow(array[i], cmap=cm, vmin=scale[0], vmax=scale[1])
                plt.colorbar()
                if title != None:
                    plt.title(title[i])
        plt.tight_layout()
    
    if display == True:
        plt.show()
    else:
        plt.savefig('img/' + name + '.png')

    print('Done')

