import numpy as np
from scipy.io import loadmat,savemat
from keras.datasets import mnist,fashion_mnist,cifar10
import os

lenet_flag = True

num_sparse1 = loadmat('/home/nagarjun/datasets/masks/sparse_pixels_gray.mat')
num_sparse1 = num_sparse1['sparse_pixels']
num_sparse1 = num_sparse1.astype(int)
num_sparse1 = num_sparse1[0]


num_sparse2 = loadmat('/home/nagarjun/datasets/svhn_resized/sparse_pixels_rgb.mat')
num_sparse2 = num_sparse2['sparse_pixels']
num_sparse2 = num_sparse2.astype(int)
num_sparse2 = num_sparse2[0]

def load_mask(x):

    if x == 'mnist':
        mnist_mask = loadmat('/home/nagarjun/datasets/masks/mnist/mask_mnist_cnn.mat')
        mnist_mask = mnist_mask['mask']
        return mnist_mask
    else:
        if x == 'fashion_mnist':
            fmnist = loadmat('/home/nagarjun/datasets/masks/fashion_mnist/mask_fashion_mnist_cnn.mat')
            fmnist = fmnist['mask']
            return fmnist
        else:
            if x == 'nist_char':

                nist = loadmat('/home/nagarjun/datasets/masks/nist/mask_nist_char_cnn.mat')
                nist = nist['mask']
                return nist
            else:
                if x == 'cifar10':

                    if lenet_flag == True:

                        cifar1 = loadmat('/home/nagarjun/datasets/masks/cifar10/lenet/mask1.mat')
                        cifar1 = cifar1['mask1']
                        print(cifar1.shape)

                        cifar2 = loadmat('/home/nagarjun/datasets/masks/cifar10/lenet/mask2.mat')
                        cifar2 = cifar2['mask2']

                        cifar3 = loadmat('/home/nagarjun/datasets/masks/cifar10/lenet/mask3.mat')
                        cifar3 = cifar3['mask3']
                        return (cifar1, cifar2, cifar3)
                    else:

                        cifar1 = loadmat('/home/nagarjun/datasets/masks/cifar10/mask_cifar_red_resized.mat')
                        cifar1 = cifar1['mask1']

                        cifar2 = loadmat('/home/nagarjun/datasets/masks/cifar10/mask_cifar_green_resized.mat')
                        cifar2 = cifar2['mask2']

                        cifar3 = loadmat('/home/nagarjun/datasets/masks/cifar10/mask_cifar_blue_resized.mat')
                        cifar3 = cifar3['mask3']
                        return (cifar1,cifar2,cifar3)
                else:
                    if x == 'svhn':

                        if lenet_flag == True:

                            svhn1 = loadmat('/home/nagarjun/datasets/masks/svhn/lenet/mask1.mat')
                            svhn1 = svhn1['mask1']

                            svhn2 = loadmat('/home/nagarjun/datasets/masks/svhn/lenet/mask2.mat')
                            svhn2 = svhn2['mask2']

                            svhn3 = loadmat('/home/nagarjun/datasets/masks/svhn/lenet/mask3.mat')
                            svhn3 = svhn3['mask3']
                            return (svhn1, svhn2, svhn3)

                        else:
                            svhn1 = loadmat('/home/nagarjun/datasets/masks/svhn/mask_svhn_red_resized.mat')
                            svhn1 = svhn1['mask1']

                            svhn2 = loadmat('/home/nagarjun/datasets/masks/svhn/mask_svhn_green_resized.mat')
                            svhn2 = svhn2['mask2']

                            svhn3 = loadmat('/home/nagarjun/datasets/masks/svhn/mask_svhn_blue_resized.mat')
                            svhn3 = svhn3['mask3']
                            return (svhn1,svhn2,svhn3)


def load_input(x):

    if x == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (x_train, y_train), (x_test, y_test)
    else:
        if x == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            # x_train = loadmat('/home/nagarjun/datasets/cifar_resized/xtrain.mat')
            # x_test = loadmat('/home/nagarjun/datasets/cifar_resized/xtest.mat')
            # x_train = x_train['xtrain']
            # x_test = x_test['xtest']
            # print(np.shape(x_train))
            return (x_train, y_train), (x_test, y_test)

        else:
            if x == 'fashion_mnist':
                    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
                    return (x_train.copy(), y_train), (x_test.copy(), y_test)
            else:
                if x == 'svhn':
                    x_train = loadmat('/home/nagarjun/datasets/svhn_cnn/x_train.mat')
                    x_test = loadmat('/home/nagarjun/datasets/svhn_cnn/x_test.mat')
                    y_train = loadmat('/home/nagarjun/datasets/svhn_cnn/y_train.mat')
                    y_test = loadmat('/home/nagarjun/datasets/svhn_cnn/y_test.mat')
                    x_train = x_train['x_train']
                    x_test = x_test['x_test']
                    y_train = y_train['y_train']
                    y_test = y_test['y_test']
                    return (x_train, y_train), (x_test, y_test)
                else:
                    if x == 'nist_char':
                            x_train = loadmat('/home/nagarjun/datasets/nist_char_cnn/x_train.mat')
                            x_test = loadmat('/home/nagarjun/datasets/nist_char_cnn/x_test.mat')
                            y_train = loadmat('/home/nagarjun/datasets/nist_char_cnn/y_train.mat')
                            y_test = loadmat('/home/nagarjun/datasets/nist_char_cnn/y_test.mat')
                            x_train = x_train['x_train']
                            x_test = x_test['x_test']
                            y_train = y_train['y_train']
                            y_test = y_test['y_test']
                            return (x_train, y_train - 1), (x_test, y_test - 1)

keys_train = ['x_train_sparse1.mat','x_train_sparse2.mat','x_train_sparse3.mat','x_train_sparse4.mat','x_train_sparse5.mat','x_train_sparse6.mat','x_train_sparse7.mat','x_train_sparse8.mat','x_train_sparse9.mat','x_train_sparse10.mat','x_train_sparse11.mat','x_train_sparse12.mat','x_train_sparse13.mat','x_train_sparse14.mat','x_train_sparse15.mat','x_train_sparse16.mat','x_train_sparse17.mat','x_train_sparse18.mat']
keys_test = ['x_test_sparse1.mat','x_test_sparse2.mat','x_test_sparse3.mat','x_test_sparse4.mat','x_test_sparse5.mat','x_test_sparse6.mat','x_test_sparse7.mat','x_test_sparse8.mat','x_test_sparse9.mat','x_test_sparse10.mat','x_test_sparse11.mat','x_test_sparse12.mat','x_test_sparse13.mat','x_test_sparse14.mat','x_test_sparse15.mat','x_test_sparse16.mat','x_test_sparse17.mat','x_test_sparse18.mat']

def generate_sparse_input_2d(x):

    path1 = '/home/nagarjun/datasets/masks/mnist/sparse_train'
    path2 = '/home/nagarjun/datasets/masks/mnist/sparse_test'

    path3 = '/home/nagarjun/datasets/masks/fashion_mnist/sparse_train'
    path4 = '/home/nagarjun/datasets/masks/fashion_mnist/sparse_test'

    path5 = '/home/nagarjun/datasets/masks/nist/sparse_train'
    path6 = '/home/nagarjun/datasets/masks/nist/sparse_test'


    (x_train, y_train), (x_test, y_test) = load_input(x)

    [trainsamples,a,b] = np.shape(x_train)
    [testsamples,_,_] = np.shape(x_test)

    x_train = np.reshape(x_train,[trainsamples,a*b])
    x_test = np.reshape(x_test,[testsamples,a*b])


    # x_train = np.array(x_train)
    # x_test = np.array(x_test)

    # x_train = x_train.copy()
    # x_test = x_test.copy()

    # x_train_sparse = x_train
    # x_test_sparse = x_test

    mask = load_mask(x)

    [_,n] = np.shape(mask)

    mask = mask[0:]

    indices = mask[1:]
    indices = indices.astype(int)
    indices = indices[0]

    #print(indices)

    num_start = n-num_sparse1
    num_start = num_start.astype(int)

    #print(num_start)

    #print(len(num_sparse1))


    for i in range(len(num_sparse1)):

        #print(num_start[i])

        zero_values = indices[num_start[i]:n]
        zero_values = zero_values -1
        # print(zero_values,'\n')
        # print(len(zero_values))

        x_train[:, zero_values] = 0
        x_test[:, zero_values] = 0

        print('Sparse-Train', i+1, '\n', np.shape(x_train))
        print('Sparse-Test', i+1, '\n', np.shape(x_test))

        if x == 'mnist':

            savemat(os.path.join(path1, keys_train[i]),{'x_train_sparse': x_train})

            savemat(os.path.join(path2,keys_test[i]),{'x_test_sparse':x_test})
        else:
            if x == 'fashion_mnist':
                savemat(os.path.join(path3, keys_train[i]), {'x_train_sparse': x_train})

                savemat(os.path.join(path4, keys_test[i]), {'x_test_sparse': x_test})
            else:
                if x == 'nist_char':
                    savemat(os.path.join(path5, keys_train[i]), {'x_train_sparse': x_train})

                    savemat(os.path.join(path6, keys_test[i]), {'x_test_sparse': x_test})


def generate_sparse_input_3d(x):

    path1 = '/home/nagarjun/datasets/masks/cifar10/lenet/sparse_train'
    path2 = '/home/nagarjun/datasets/masks/cifar10/lenet/sparse_test'

    path3 = '/home/nagarjun/datasets/masks/svhn/lenet/sparse_train'
    path4 = '/home/nagarjun/datasets/masks/svhn/lenet/sparse_test'

    (x_train, y_train), (x_test, y_test) = load_input(x)

    [trainsamples, a, b,_] = np.shape(x_train)
    [testsamples, _, _,_] = np.shape(x_test)


    x_train1 = x_train[:,:,:,0]
    x_train2 = x_train[:,:,:,1]
    x_train3 = x_train[:,:,:,2]

    x_test1 = x_test[:,:,:,0]
    x_test2 = x_test[:,:,:,1]
    x_test3 = x_test[:,:,:,2]

    x_train1 = np.reshape(x_train1,[trainsamples,a*b])
    x_train2 = np.reshape(x_train2,[trainsamples,a*b])
    x_train3 = np.reshape(x_train3,[trainsamples,a*b])

    x_test1 = np.reshape(x_test1, [testsamples, a*b])
    x_test2 = np.reshape(x_test2, [testsamples, a*b])
    x_test3 = np.reshape(x_test3, [testsamples, a*b])

    [mask1,mask2,mask3] = load_mask(x)

    if lenet_flag == True:

        indices1 = mask1
        indices2 = mask2
        indices3 = mask3

        [n,_] = np.shape(mask1)

        num_start = n - num_sparse2

        for i in range(len(num_sparse2)):

            zero_values1 = indices1[num_start[i]:n]
            zero_values2 = indices2[num_start[i]:n]
            zero_values3 = indices3[num_start[i]:n]

            zero_values1 = zero_values1 - 1
            zero_values2 = zero_values2 - 1
            zero_values3 = zero_values3 - 1

            # print(zero_values1,'\n')
            # print(len(zero_values))

            x_train1[:, zero_values1] = 0
            x_test1[:, zero_values1] = 0

            x_train2[:, zero_values2] = 0
            x_test2[:, zero_values2] = 0

            x_train3[:, zero_values3] = 0
            x_test3[:, zero_values3] = 0

            # x_train1 = np.reshape(x_train1,[trainsamples,img_x,img_x])
            # x_train2 = np.reshape(x_train2, [trainsamples, img_x, img_x])
            # x_train3 = np.reshape(x_train3, [trainsamples, img_x, img_x])
            #
            # x_test1 = np.reshape(x_test1, [testsamples, img_x, img_x])
            # x_test2 = np.reshape(x_test2, [testsamples, img_x, img_x])
            # x_test3 = np.reshape(x_test3, [testsamples, img_x, img_x])

            x_train_sparse = np.dstack((x_train1, x_train2, x_train3))
            x_test_sparse = np.dstack((x_test1, x_test2, x_test3))

            print('Sparse-Train', i + 1, '\n', np.shape(x_train_sparse))
            print('Sparse-Test', i + 1, '\n', np.shape(x_test_sparse))

            if x == 'cifar10':
                savemat(os.path.join(path1, keys_train[i]), {'x_train_sparse': x_train_sparse})

                savemat(os.path.join(path2, keys_test[i]), {'x_test_sparse': x_test_sparse})
            else:
                if x == 'svhn':
                    savemat(os.path.join(path3, keys_train[i]), {'x_train_sparse': x_train_sparse})

                    savemat(os.path.join(path4, keys_test[i]), {'x_test_sparse': x_test_sparse})

    else:
        mask1 = mask1[0:]
        indices1 = mask1[1:]
        indices1 = indices1.astype(int)
        indices1 = indices1[0]

        mask2 = mask2[0:]
        indices2 = mask2[1:]
        indices2 = indices2.astype(int)
        indices2 = indices2[0]

        mask3 = mask3[0:]
        indices3 = mask3[1:]
        indices3 = indices3.astype(int)
        indices3 = indices3[0]

        [m, n] = np.shape(mask1)

        num_start = n - num_sparse2
        num_start = num_start.astype(int)

        for i in range(len(num_sparse2)):

            zero_values1 = indices1[num_start[i]:n]
            zero_values2 = indices2[num_start[i]:n]
            zero_values3 = indices3[num_start[i]:n]

            zero_values1 = zero_values1 - 1
            zero_values2 = zero_values2 - 1
            zero_values3 = zero_values3 - 1

            #print(zero_values1,'\n')
            # print(len(zero_values))

            x_train1[:, zero_values1] = 0
            x_test1[:, zero_values1] = 0

            x_train2[:, zero_values2] = 0
            x_test2[:, zero_values2] = 0

            x_train3[:, zero_values3] = 0
            x_test3[:, zero_values3] = 0

            # x_train1 = np.reshape(x_train1,[trainsamples,img_x,img_x])
            # x_train2 = np.reshape(x_train2, [trainsamples, img_x, img_x])
            # x_train3 = np.reshape(x_train3, [trainsamples, img_x, img_x])
            #
            # x_test1 = np.reshape(x_test1, [testsamples, img_x, img_x])
            # x_test2 = np.reshape(x_test2, [testsamples, img_x, img_x])
            # x_test3 = np.reshape(x_test3, [testsamples, img_x, img_x])

            x_train_sparse = np.dstack((x_train1,x_train2,x_train3))
            x_test_sparse = np.dstack((x_test1,x_test2,x_test3))

            print('Sparse-Train',i+1,'\n',np.shape(x_train_sparse))
            print('Sparse-Test',i+1,'\n',np.shape(x_test_sparse))

            if x == 'cifar10':

                savemat(os.path.join(path1, keys_train[i]), {'x_train_sparse': x_train_sparse})

                savemat(os.path.join(path2, keys_test[i]), {'x_test_sparse': x_test_sparse})
            else:
                if x == 'svhn':

                    savemat(os.path.join(path3, keys_train[i]), {'x_train_sparse': x_train_sparse})

                    savemat(os.path.join(path4, keys_test[i]), {'x_test_sparse': x_test_sparse})



#generate_sparse_input_2d('nist_char')
generate_sparse_input_3d('svhn')

