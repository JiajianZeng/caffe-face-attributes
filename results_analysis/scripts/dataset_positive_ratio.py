from matplotlib import pyplot as plt
import numpy as np
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Positive ratios of datasets')
    parser.add_argument('--celeba_ratio_file', dest='celeba_positive_ratio_file',
                        help='positive ratios of celeba',
                        default=None, type=str)
    parser.add_argument('--lfwa_ratio_file', dest='lfwa_positive_ratio_file',
                        help='positive ratios of lfwa',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def bar_plot_positive_ratio(celeba_positive_ratio_file, lfwa_positive_ratio_file):
    face_attributes_name = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                            'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                            'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                            'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                            'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                            'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                            'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    # get positive ratios of different attributes from celeba
    with open(celeba_positive_ratio_file, 'r') as f:
        lines = f.readlines()
    ratio_positive_celeba = [float(x) for x in lines]
    print ratio_positive_celeba

    # get positive ratios of different attributes from lfwa
    with open(lfwa_positive_ratio_file, 'r') as f:
        lines = f.readlines()
    ratio_positive_lfwa = [float(x) for x in lines]
    print ratio_positive_lfwa

    # bar plot
    fig = plt.figure()
    # get subplot, 111 means split the figure into 1 (rows) * 1 (ncols) sub-axes
    ax = fig.add_subplot(111)
    ax.axhline(y = 0.1, color = 'red', linestyle = 'dashed')
    ax.axhline(y = 0.3, color = 'red', linestyle = 'dashed')
    # bar width
    width = 0.35
    left = np.arange(len(face_attributes_name))

    # bars
    bars_celeba = ax.bar(left, ratio_positive_celeba, width,
                    color='red')
    bars_lfwa = ax.bar(left + width, ratio_positive_lfwa, width,
                  color='green')

    # axes and labels
    ax.set_xlim(-width, len(left) + width)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Positive sample ratio',fontsize=15)
    # ax.set_title('Ratio of positive samples for each category of attributes')
    ax.set_xticks(left + width)
    ax.set_xticklabels(face_attributes_name, rotation=90, fontsize=15)

    # add a legend
    ax.legend((bars_celeba[0], bars_lfwa[0]), ('CelebA', 'LFWA'), fontsize=15)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    bar_plot_positive_ratio(args.celeba_positive_ratio_file, args.lfwa_positive_ratio_file)