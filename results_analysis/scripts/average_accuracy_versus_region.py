import os

def bar_average_acc():
    region_name = ['Eye Region', 'Nose Region', 'Mouth Region', 'Cheek Region',
                   'Hair Region', 'Neck Region', 'Global']

    # average acc for celeba
    celeba_acc_facetracer = [82.40, 71.00, 82.75, 86.14,
                             77.80, 80.67, 81.89]
    celeba_acc_panda_w = [78.60, 67.50, 78.50, 82.57,
                          80.90, 80.00, 80.56]
    celeba_acc_panda_1 = [85.00, 73.00, 80.00, 88.29,
                          86.40, 82.00, 85.78]
    celeba_acc_lnets_anet = [85.60, 75.00, 87.00, 90.86,
                             89.40, 85.33, 86.67]
    celeba_acc_our_approach = [89.20, 80.50, 88.75, 94.14,
                               92.80, 93.67, 90.33]

    # average acc for lfwa
    lfwa_acc_facetracer = [72.40, 73.50, 78.75, 73.43,
                           72.00, 74.00, 75.22]
    lfwa_acc_panda_w = [68.20, 69.50, 74.50, 69.14,
                        73.80, 71.00, 73.11]
    lfwa_acc_panda_1 = [76.00, 77.50, 82.75, 80.14,
                        81.80, 80.00, 81.78]
    lfwa_acc_lnets_anet = [84.60, 80.50, 86.00, 82.57,
                           84.90, 81.67, 83.78]
    lfwa_acc_our_approach = [84.40, 83.00, 87.25, 84.86,
                             87.20, 83.00, 85.78]
    # write celeba results
    with open(os.path.join('./results', 'average_accuracy_versus_region_celeba.txt'), 'w') as f:
        f.write('average accuracy versus region for celeba\n')
        f.write('average accuracy for global attributes in CelebA are:\n')
        f.write('CelebA: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
            celeba_acc_facetracer[6],
            celeba_acc_panda_w[6],
            celeba_acc_panda_1[6],
            celeba_acc_lnets_anet[6],
            celeba_acc_our_approach[6]
        ))

        f.write('\\hline \n')
        f.write('& {\\bf FaceTracer} & {\\bf PANDA-w} & {\\bf PANDA-1} & {\\bf LNets+ANet} & {\\bf Our approach} \\\\ \n')
        for i in xrange(6):
            f.write('\\hline\n')
            f.write('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \n'.format(
                region_name[i],
                celeba_acc_facetracer[i],
                celeba_acc_panda_w[i],
                celeba_acc_panda_1[i],
                celeba_acc_lnets_anet[i],
                celeba_acc_our_approach[i]
            ))
        f.write('\\hline\n')

    # write lfwa results

    with open(os.path.join('./results', 'average_accuracy_versus_region_lfwa.txt'), 'w') as f:
        f.write('average accuracy versus region for lfwa\n')
        f.write('average accuracy for global attributes in LFW are:\n')
        f.write('LFW: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
            lfwa_acc_facetracer[6],
            lfwa_acc_panda_w[6],
            lfwa_acc_panda_1[6],
            lfwa_acc_lnets_anet[6],
            lfwa_acc_our_approach[6]
        ))

        f.write('\\hline\n')
        f.write(
            '& {\\bf FaceTracer} & {\\bf PANDA-w} & {\\bf PANDA-1} & {\\bf LNets+ANet} & {\\bf Our approach} \\\\ \n')
        for i in xrange(6):
            f.write('\\hline\n')
            f.write('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \n'.format(
                region_name[i],
                lfwa_acc_facetracer[i],
                lfwa_acc_panda_w[i],
                lfwa_acc_panda_1[i],
                lfwa_acc_lnets_anet[i],
                lfwa_acc_our_approach[i]
            ))
        f.write('\\hline\n')


    '''width = 0.35
    left = np.arange(len(region_name)) * 6 * width

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    # bars
    bars_celeba_facetracer = ax.bar(left + 1, celeba_acc_facetracer, width,
                                    color='red')
    bars_celeba_panda_w = ax.bar(left + 1 + width, celeba_acc_panda_w, width,
                                 color='blue')
    bars_celeba_panda_1 = ax.bar(left + 1 + 2 * width, celeba_acc_panda_1, width,
                                 color='green')
    bars_celeba_lnets_anet = ax.bar(left + 1 + 3 * width, celeba_acc_lnets_anet, width,
                                    color='black')
    bars_celeba_our_approach = ax.bar(left + 1 + 4 * width, celeba_acc_our_approach, width,
                                      color='yellow')

    # axes
    ax.set_xlim(0,  1 + 49 * width)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Average Accuracy')
    ax.set_xticks(left + 1 + 2.5 * width)
    ax.set_xticklabels(region_name)

    ax.legend((bars_celeba_facetracer[0], bars_celeba_panda_w[0], bars_celeba_panda_1[0], bars_celeba_lnets_anet[0], bars_celeba_our_approach[0]),
              ('FaceTracer', 'PANDA-w', 'PANDA-1', 'LNets+ANet', 'Our appoach'))'''

    # plot for celeba
    '''plt.figure(1)
    left = np.arange(len(region_name))
    axis = [0, len(left) + 1, 50, 100]
    # performance of FaceTracer plot
    plt.plot(left + 1, celeba_acc_facetracer, 'r^--', label='FaceTracer')
    # performance of PANDA-w plot
    plt.plot(left + 1, celeba_acc_panda_w, 'bo--', label='PANDA-w')
    # performance of PANDA-1 plot
    plt.plot(left + 1, celeba_acc_panda_1, 'g*--', label='PANDA-1')
    # performance of LNets+Anet
    plt.plot(left + 1, celeba_acc_lnets_anet, 'k+--', label='LNets+ANet')
    # performance of our appraoch
    plt.plot(left + 1, celeba_acc_our_approach, 'ys--', label='Our approach')
    # axes and labels
    plt.axis(axis)
    plt.ylabel('Average Accuracy',fontsize=15)
    # plt.title('Accuracy of attributes with low positive sample ratio')

    # legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode='expand', borderaxespad=0, fontsize=15)
    plt.xticks(left + 1, region_name, fontsize=15)

    # plot for lfwa
    plt.figure(2)
    left = np.arange(len(region_name))
    axis = [0, len(left) + 1, 50, 100]
    # performance of FaceTracer plot
    plt.plot(left + 1, lfwa_acc_facetracer, 'r^--', label='FaceTracer')
    # performance of PANDA-w plot
    plt.plot(left + 1, lfwa_acc_panda_w, 'bo--', label='PANDA-w')
    # performance of PANDA-1 plot
    plt.plot(left + 1, lfwa_acc_panda_1, 'g*--', label='PANDA-1')
    # performance of LNets+Anet
    plt.plot(left + 1, lfwa_acc_lnets_anet, 'k+--', label='LNets+ANet')
    # performance of our appraoch
    plt.plot(left + 1, lfwa_acc_our_approach, 'ys--', label='Our approach')
    # axes and labels
    plt.axis(axis)
    plt.ylabel('Average Accuracy', fontsize=15)
    # plt.title('Accuracy of attributes with low positive sample ratio')

    # legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode='expand', borderaxespad=0, fontsize=15)
    plt.xticks(left + 1, region_name, fontsize=15)

    plt.show()'''

if __name__ == '__main__':
    bar_average_acc()