import matplotlib.pyplot as plt


def comparison_curve(
    history: dict,
    criterion: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    plt.plot(history['trn'], label='TRN')
    plt.plot(history['val'], label='VAL')
    plt.xlabel('EPOCH')
    plt.ylabel(criterion)
    plt.title('TRAIN vs. VALIDATION')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def criterion_curve(
    history: dict,
    criterion: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    plt.plot(history['loo'], label='LOO')
    plt.xlabel('EPOCH')
    plt.ylabel(criterion)
    plt.title('LEAVE ONE OUT')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()