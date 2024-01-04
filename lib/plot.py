import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes


def _timeXaxis(ax: Axes, matrix, tpt):
    xticks = np.arange(0, matrix.shape[1], 1 / tpt)
    xticklabels = (xticks * tpt).round().astype(np.int32)
    ax.set_xticks(xticks, xticklabels)
    plt.xlabel("Time (s)")


def _tokensYaxis(ax: Axes, words, word_tokens):
    ylims = ax.get_ylim()
    ax.tick_params("both", length=0, width=0, which="minor", pad=6)
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_position("left")
    ax.invert_yaxis()
    ax.set_ylim(ylims)
    major_ticks = [-0.5]
    minor_ticks = []
    current_y = 0
    for word, word_token in zip(words, word_tokens):
        minor_ticks.append(current_y + len(word_token) / 2 - 0.5)
        current_y += len(word_token)
        major_ticks.append(current_y - 0.5)
    ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(words))
    ax.set_yticks(major_ticks)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    plt.ylabel("Tokens")


def plotHead(title, matrix, alignment, words, word_tokens, tpt):
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, aspect="auto")
    plt.plot(alignment.index2s, alignment.index1s, color="red")
    ax = plt.gca()
    ax.set_title(title)
    _timeXaxis(ax, matrix, tpt)
    _tokensYaxis(ax, words, word_tokens)
    plt.show()
