import numpy as np
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Low positive ratio versus performance')
    parser.add_argument('--celeba_ratio_performance_file', dest='celeba_ratio_performance_file',
                        help='ratio versus performance of celeba',
                        default=None, type=str)
    parser.add_argument('--lfwa_ratio_performance_file', dest='lfwa_ratio_performance_file',
                        help='ratio versus performance of lfwa',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def bar_plot_positive_ratio(celeba_ratio_performance_file, lfwa_ratio_performance_file):
    face_attributes_name = np.array(['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                            'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                            'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                            'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                            'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                            'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                            'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'])
    celeba_low_positive_ratio_inds = [4, 10, 13, 14,
                               15, 16, 17, 22,
                               26, 28, 29, 30, 35, 38]
    celeba_face_attributes_name = face_attributes_name[celeba_low_positive_ratio_inds]

    lfwa_low_positive_ratio_inds = [1, 4, 5, 8,
                                    9, 10, 15, 16,
                                    17, 18, 22, 29,
                                    34, 35, 36, 37, 39]

    lfwa_face_attributes_name = face_attributes_name[lfwa_low_positive_ratio_inds]

    # get positive ratios versus performace of specifical attributes from celeba
    with open(celeba_ratio_performance_file, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    ratio_positive_celeba = np.array([float(x[0]) for x in splitlines])
    celeba_acc_facetracer = np.array([float(x[1]) for x in splitlines])
    celeba_acc_panda_w = np.array([float(x[2]) for x in splitlines])
    celeba_acc_panda_1 = np.array([float(x[3]) for x in splitlines])
    celeba_acc_lnets_anet = np.array([float(x[4]) for x in splitlines])
    celeba_acc_our_approach = np.array([float(x[5]) for x in splitlines])

    # filter low positive attributes
    ratio_positive_celeba = ratio_positive_celeba[celeba_low_positive_ratio_inds]
    celeba_acc_facetracer = celeba_acc_facetracer[celeba_low_positive_ratio_inds]
    celeba_acc_panda_w = celeba_acc_panda_w[celeba_low_positive_ratio_inds]
    celeba_acc_panda_1 = celeba_acc_panda_1[celeba_low_positive_ratio_inds]
    celeba_acc_lnets_anet = celeba_acc_lnets_anet[celeba_low_positive_ratio_inds]
    celeba_acc_our_approach = celeba_acc_our_approach[celeba_low_positive_ratio_inds]

    # write celeba results
    with open(os.path.join('./results', 'acc_for_low_positive_sample_ratio_celeba.txt'), 'w') as f:
        f.write('accuracy for low positive sample ratio for celeba\n')
        f.write('positive ratios for each attribute are:\n')
        for i in xrange(len(ratio_positive_celeba)):
            f.write('{:.2f} '.format(ratio_positive_celeba[i]))
        f.write('\n')

        f.write('\\hline\n')
        f.write('&{\\bf FaceTracer} & {\\bf PANDA-w} & {\\bf PANDA-1} & {\\bf LNets+ANet} & {\\bf Our approach} \\\\ \n')
        for i in xrange(len(celeba_low_positive_ratio_inds)):
            f.write('\\hline\n')
            f.write('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \n'.format(
                celeba_face_attributes_name[i],
                celeba_acc_facetracer[i],
                celeba_acc_panda_w[i],
                celeba_acc_panda_1[i],
                celeba_acc_lnets_anet[i],
                celeba_acc_our_approach[i]
            ))
        f.write('\\hline\n')

    # get positive ratios versus performace of specifical attributes from lfwa
    with open(lfwa_ratio_performance_file, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    ratio_positive_lfwa = np.array([float(x[0]) for x in splitlines])
    lfwa_acc_facetracer = np.array([float(x[1]) for x in splitlines])
    lfwa_acc_panda_w = np.array([float(x[2]) for x in splitlines])
    lfwa_acc_panda_1 = np.array([float(x[3]) for x in splitlines])
    lfwa_acc_lnets_anet = np.array([float(x[4]) for x in splitlines])
    lfwa_acc_our_approach = np.array([float(x[5]) for x in splitlines])

    # filter low positive attributes
    ratio_positive_lfwa = ratio_positive_lfwa[lfwa_low_positive_ratio_inds]
    lfwa_acc_facetracer = lfwa_acc_facetracer[lfwa_low_positive_ratio_inds]
    lfwa_acc_panda_w = lfwa_acc_panda_w[lfwa_low_positive_ratio_inds]
    lfwa_acc_panda_1 = lfwa_acc_panda_1[lfwa_low_positive_ratio_inds]
    lfwa_acc_lnets_anet = lfwa_acc_lnets_anet[lfwa_low_positive_ratio_inds]
    lfwa_acc_our_approach = lfwa_acc_our_approach[lfwa_low_positive_ratio_inds]

    # write lfwa results
    # write celeba results
    with open(os.path.join('./results', 'acc_for_low_positive_sample_ratio_lfwa.txt'), 'w') as f:
        f.write('accuracy for low positive sample ratio for lfwa\n')
        f.write('positive ratios for each attribute are:\n')
        for i in xrange(len(ratio_positive_lfwa)):
            f.write('{:.2f} '.format(ratio_positive_lfwa[i]))
        f.write('\n')

        f.write('\\hline\n')
        f.write(
            '&{\\bf FaceTracer} & {\\bf PANDA-w} & {\\bf PANDA-1} & {\\bf LNets+ANet} & {\\bf Our approach} \\\\ \n')
        for i in xrange(len(lfwa_low_positive_ratio_inds)):
            f.write('\\hline\n')
            f.write('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \n'.format(
                lfwa_face_attributes_name[i],
                lfwa_acc_facetracer[i],
                lfwa_acc_panda_w[i],
                lfwa_acc_panda_1[i],
                lfwa_acc_lnets_anet[i],
                lfwa_acc_our_approach[i]
            ))
        f.write('\\hline\n')

    # plot for celeba
    '''plt.figure(1)
    left = np.arange(len(celeba_face_attributes_name))
    axis = [0, len(left) + 1, 0, 1]
    # performance of FaceTracer plot
    plt.plot(left + 1, celeba_acc_facetracer * 0.01, 'r^--', label = 'FaceTracer')
    # performance of PANDA-w plot
    plt.plot(left + 1, celeba_acc_panda_w * 0.01, 'bo--', label = 'PANDA-w')
    # performance of PANDA-1 plot
    plt.plot(left + 1, celeba_acc_panda_1 * 0.01, 'g*--', label = 'PANDA-1')
    # performance of LNets+Anet
    plt.plot(left + 1, celeba_acc_lnets_anet * 0.01, 'k+--', label = 'LNets+ANet')
    # performance of our appraoch
    plt.plot(left + 1, celeba_acc_our_approach * 0.01, 'ys--', label = 'Our approach')
    # axes and labels
    plt.axis(axis)
    plt.ylabel('Accuracy', fontsize=15)
    # plt.title('Accuracy of attributes with low positive sample ratio')

    # legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode='expand', borderaxespad=0, fontsize=15)
    plt.xticks(left + 1, celeba_face_attributes_name, rotation=90, fontsize=15)

    # bar plot
    ax_bar_celeba = plt.twinx()
    ax_bar_celeba.set_ylabel('Positive sample ratio',fontsize=15)
    ax_bar_celeba.set_ylim(0, 0.2)
    ax_bar_celeba.bar(left + 1, ratio_positive_celeba, 0.35,
            color = 'red',
            align = 'center')



    # plot for lfwa
    plt.figure(2)
    left_lfwa = np.arange(len(lfwa_face_attributes_name))
    axis_lfwa = [0, len(left_lfwa) + 1, 0, 1]
    # performance of FaceTracer plot
    plt.plot(left_lfwa + 1, lfwa_acc_facetracer * 0.01, 'r^--', label='FaceTracer')
    # performance of PANDA-w plot
    plt.plot(left_lfwa + 1, lfwa_acc_panda_w * 0.01, 'bo--', label='PANDA-w')
    # performance of PANDA-1 plot
    plt.plot(left_lfwa + 1, lfwa_acc_panda_1 * 0.01, 'g*--', label='PANDA-1')
    # performance of LNets+Anet
    plt.plot(left_lfwa + 1, lfwa_acc_lnets_anet * 0.01, 'k+--', label='LNets+ANet')
    # performance of our appraoch
    plt.plot(left_lfwa + 1, lfwa_acc_our_approach * 0.01, 'ys--', label='Our approach')
    # axes and labels
    plt.axis(axis_lfwa)
    plt.ylabel('Accuracy', fontsize=15)
    # plt.title('Accuracy of attributes with low positive sample ratio')

    # legend
    plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3, ncol = 5, mode = 'expand', borderaxespad = 0, fontsize=15)
    plt.xticks(left_lfwa + 1, lfwa_face_attributes_name, rotation=90, fontsize=15)

    # bar plot
    ax_bar_lfwa = plt.twinx()
    ax_bar_lfwa.set_ylabel('Positive sample ratio', fontsize=15)
    ax_bar_lfwa.set_ylim(0, 0.5)
    ax_bar_lfwa.bar(left_lfwa + 1, ratio_positive_lfwa, 0.35,
            color='red',
            align='center')
    # ax_bar_lfwa.legend((bars_lfwa[0],),('Ratio of positive samples',))
    plt.show()'''


if __name__ == '__main__':
    args = parse_args()
    bar_plot_positive_ratio(args.celeba_ratio_performance_file, args.lfwa_ratio_performance_file)