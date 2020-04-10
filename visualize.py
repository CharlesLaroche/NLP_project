import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    import skimage.transform
    import matplotlib.cm as cm

    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    f, axs = plt.subplots(np.int(np.ceil(len(words) / 5.)), 5, figsize=(15, 10))
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0)

    for t in range(len(words)):
        if t > 50:
            break
        axs[t // 5, t % 5].text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        axs[t // 5, t % 5].imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            axs[t // 5, t % 5].imshow(alpha, alpha=0, cmap=cm.Greys_r)
        else:
            axs[t // 5, t % 5].imshow(alpha, alpha=0.8, cmap=cm.Greys_r)
        axs[t // 5, t % 5].axis('off')

    for t in range(np.int(np.ceil(len(words) / 5.)) * 5):
        axs[t // 5, t % 5].axis('off')
    plt.savefig("ATT.jpg")
    plt.show()
