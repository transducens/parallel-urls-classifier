
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_confusion_matrix(outputs_argmax, labels, classes=2):
    tp, fp, fn, tn = np.zeros(classes), np.zeros(classes), np.zeros(classes), np.zeros(classes)

    for c in range(classes):
        # Multiclass confusion matrix
        # https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
        tp[c] = torch.sum(torch.logical_and(labels == c, outputs_argmax == c))
        fp[c] = torch.sum(torch.logical_and(labels != c, outputs_argmax == c))
        fn[c] = torch.sum(torch.logical_and(labels == c, outputs_argmax != c))
        tn[c] = torch.sum(torch.logical_and(labels != c, outputs_argmax != c))

    return {"tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,}

def get_metrics(outputs_argmax, labels, current_batch_size, logger, classes=2, idx=-1, log=False):
    acc = (torch.sum(outputs_argmax == labels) / current_batch_size).cpu().detach().numpy()

    no_values_per_class = np.zeros(classes)
    acc_per_class = np.zeros(classes)
    precision, recall, f1 = np.zeros(classes), np.zeros(classes), np.zeros(classes)
    macro_f1 = 0.0

    conf_mat = get_confusion_matrix(outputs_argmax, labels, classes=classes)
    tp, fp, fn, tn = conf_mat["tp"], conf_mat["fp"], conf_mat["fn"], conf_mat["tn"]

    for c in range(classes):
        no_values_per_class[c] = torch.sum(labels == c)

        # How many times have we classify correctly the target class taking into account all the data? -> we get how many percentage is from each class
        acc_per_class[c] = torch.sum(torch.logical_and(labels == c, outputs_argmax == c)) / current_batch_size

        # Metrics
        # http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf
        # https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
        precision[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) != 0 else 1.0
        recall[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) != 0 else 1.0
        f1[c] = 2 * ((precision[c] * recall[c]) / (precision[c] + recall[c])) if not np.isclose(precision[c] + recall[c], 0.0) else 0.0

    #assert outputs.shape[-1] == acc_per_class.shape[-1], f"Shape of outputs does not match the acc per class shape ({outputs.shape[-1]} vs {acc_per_class.shape[-1]})"
    assert np.isclose(np.sum(acc_per_class), acc), f"Acc and the sum of acc per classes should match ({acc} vs {np.sum(acc_per_class)})"

    macro_f1 = np.sum(f1) / f1.shape[0]

    if log:
        logger.debug("[train:batch#%d] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)", idx + 1, acc * 100.0, acc_per_class[0] * 100.0, acc_per_class[1] * 100.0)
        logger.debug("[train:batch#%d] Acc per class (non-parallel->precision|recall|f1, parallel->precision|recall|f1): (%d -> %.2f %% | %.2f %% | %.2f %%, %d -> %.2f %% | %.2f %% | %.2f %%)",
                     idx + 1, no_values_per_class[0], precision[0] * 100.0, recall[0] * 100.0, f1[0] * 100.0, no_values_per_class[1], precision[1] * 100.0, recall[1] * 100.0, f1[1] * 100.0)
        logger.debug("[train:batch#%d] Macro F1: %.2f %%", idx + 1, macro_f1 * 100.0)

    return {"acc": acc,
            "acc_per_class": acc_per_class,
            "no_values_per_class": no_values_per_class,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": macro_f1,}

def plot_statistics(args, path=None, time_wait=5.0, freeze=False):
    plt_plot_common_params = {"marker": 'o', "markersize": 2,}
    plt_scatter_common_params = {"marker": 'o', "s": 2,}
    plt_legend_common_params = {"loc": "center left", "bbox_to_anchor": (1, 0.5), "fontsize": "x-small",}

    plt.clf()

    plt.subplot(3, 2, 1)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_loss"]))))), args["batch_loss"], label="Train loss", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 5)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc"]))))), args["batch_acc"], label="Train acc", **plt_plot_common_params)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc_classes"][0]))))), args["batch_acc_classes"][0], label="Train F1: no p.", **plt_plot_common_params)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc_classes"][1]))))), args["batch_acc_classes"][1], label="Train F1: para.", **plt_plot_common_params)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_macro_f1"]))))), args["batch_macro_f1"], label="Train macro F1", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 2)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_loss"], label="Train loss", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_loss"], label="Dev loss", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 3)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_acc"], label="Train acc", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_acc_classes"][0], label="Train F1: no p.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_acc_classes"][1], label="Train F1: para.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_macro_f1"], label="Train macro F1", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 4)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_acc"], label="Dev acc", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_acc_classes"][0], label="Dev F1: no p.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_acc_classes"][1], label="Dev F1: para.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_macro_f1"], label="Dev macro F1", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plot_final = True if args["final_dev_acc"] else False

    plt.subplot(3, 2, 6)
    plt.scatter(0 if plot_final else None, args["final_dev_acc"] if plot_final else None, label="Dev acc", **plt_scatter_common_params)
    plt.scatter(0 if plot_final else None, args["final_test_acc"] if plot_final else None, label="Test acc", **plt_scatter_common_params)
    plt.scatter(1 if plot_final else None, args["final_dev_macro_f1"] if plot_final else None, label="Dev macro F1", **plt_scatter_common_params)
    plt.scatter(1 if plot_final else None, args["final_test_macro_f1"] if plot_final else None, label="Test macro F1", **plt_scatter_common_params)
    plt.legend(**plt_legend_common_params)

    #plt.tight_layout()
    plt.subplots_adjust(left=0.08,
                        bottom=0.07,
                        right=0.8,
                        top=0.95,
                        wspace=1.0,
                        hspace=0.4)

    if path:
        plt.savefig(path, dpi=1200)
    else:
        if freeze:
            plt.show()
        else:
            plt.pause(time_wait)
