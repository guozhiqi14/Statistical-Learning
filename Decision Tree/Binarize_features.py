#Can't import package cv2 under environment python2

'''
Module to produce binarized feature output
'''

# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.cv.CV_THRESH_BINARY_INV,cv_img)
    return cv_img

@log
def binaryzation_features(trainset):
    features = []

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        img_b = binaryzation(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(img_b)

    features = np.array(features)
    features = np.reshape(features,(-1,784))

    return features


df = pd.read_csv('/Users/guozhiqi-seven/Documents/Statistical Learning/Decision Tree/train.csv')
data = df.values

imgs = data[0::,1::]
labels = data[::,0]

features = binaryzation_features(imgs) #二值化之后， 所有值只有0 or 1
np.savetxt('features.out',features) #Big file,around 800 M