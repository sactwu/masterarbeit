import numpy as np
import matplotlib.pyplot as plt


def main():
    filename = '77_karl_carstens_1_edges_humans.csv'
    g_rr = np.load('Datasets/results/g_rr_{0}.npy'.format(filename))[:15, :15]
    g_pr = np.load('Datasets/results/g_pr_{0}.npy'.format(filename))[:15, :15]
    g_qr = np.load('Datasets/results/g_qr_{0}.npy'.format(filename))[:15, :15]
    g_r = np.load('Datasets/results/g_r_{0}.npy'.format(filename))[:15, :15]
    print('g_rr: ', g_rr)
    print('g_pr: ', g_pr)
    print('g_qr: ', g_qr)
    print('g_r: ', g_r)
    print('plotting the matrices')
    plt.matshow(g_qr)
    plt.title('g_qr')
    plt.colorbar()
    plt.savefig('Datasets/results/g_qr_{0}.png'.format(filename))
    plt.matshow(g_pr)
    plt.title('g_pr')
    plt.colorbar()
    plt.savefig('Datasets/results/g_pr_{0}.png'.format(filename))
    plt.matshow(g_rr)
    plt.title('g_rr')
    plt.colorbar()
    plt.savefig('Datasets/results/g_rr_{0}.png'.format(filename))
    plt.matshow(g_r)
    plt.title('g_r')
    plt.colorbar()
    plt.savefig('Datasets/results/g_r_{0}.png'.format(filename))
    print('finished saving the results to files')


if __name__ == '__main__':
    main()
