import numpy as np

class_list = np.load('./data/ActivityNet12/features_annotations_anet1.2/ActivityNet1.2-Annotations/classlist.npy').astype(str)

class_dict = dict()
for i, class_name in enumerate(class_list):
    class_dict[class_name] = i


all_features = np.load('./data/ActivityNet12/features_annotations_anet1.2/ActivityNet1.2-I3D-JOINTFeatures.npy', allow_pickle=True, encoding='latin1')
subset = np.load('./data/ActivityNet12/features_annotations_anet1.2/ActivityNet1.2-Annotations/subset.npy', allow_pickle=True).astype(str)
labels_all = np.load('./data/ActivityNet12/features_annotations_anet1.2/ActivityNet1.2-Annotations/labels_all.npy', allow_pickle=True)
url = np.load('./data/ActivityNet12/features_annotations_anet1.2/ActivityNet1.2-Annotations/url.npy', allow_pickle=True).astype(str)

train_labels = np.zeros((4819, 100))
test_labels = np.zeros((2383, 100))
ActivityNet12_test_vid_list = open('ActivityNet12_test_vid_list.txt', 'w')

train_num = 0
test_num = 0
for i in range(all_features.shape[0]):
    print(i, train_num, test_num)

    if subset[i] == 'training':
        np.save('./data/ActivityNet12/train_data/rgb_features/{}.npy'.format(train_num + 1),
                all_features[i][:, 1024])
        np.save('./data/ActivityNet12/train_data/flow_features/{}.npy'.format(train_num + 1),
                all_features[i][:, 1024:])
        for label in labels_all[i]:
            train_labels[train_num][class_dict[label]] = 1

        train_num += 1

    elif subset[i] == 'validation':
        np.save('./data/ActivityNet12/test_data/rgb_features/{}.npy'.format(test_num + 1),
                all_features[i][:, 1024])
        np.save('./data/ActivityNet12/test_data/flow_features/{}.npy'.format(test_num + 1),
                all_features[i][:, 1024:])
        for label in labels_all[i]:
            test_labels[test_num][class_dict[label]] = 1
        ActivityNet12_test_vid_list.write('{}\n'.format(url[i][-11:]))

        test_num += 1


print(train_num, test_num)
np.save('./train_labels.npy', train_labels)
np.save('./test_labels.npy', test_labels)
ActivityNet12_test_vid_list.close()
