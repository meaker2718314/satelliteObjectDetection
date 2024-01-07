import torch

def generate_predictions(model, loader, save_dir, device='cuda'):
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes

    model.eval()

    for idx, (imgs, targets) in enumerate(loader):
        for n, img in enumerate(imgs):
            x = img[:3, ...].to(device)

            with torch.no_grad():
                predictions = model([x, ])
                pred = predictions[0]

            x = (255.0 * (x - x.min()) / (x.max() - x.min())).to(torch.uint8)
            x = x[:3, ...]

            # Draw predicted boxes
            label_scores = list(zip(pred['labels'], pred['scores']))
            pred_labels = [f"Tank {100 * score:.0f}%" for label, score in
                           label_scores]

            pred_boxes = pred["boxes"].long()

            output_image = draw_bounding_boxes(x, pred_boxes,
                                               pred_labels, colors="green", fill=True)

            # Draw true boxes
            true_boxes = targets[n]['boxes']
            true_labels = [''] * true_boxes.size()[0]
            output_image = draw_bounding_boxes(output_image, true_boxes,
                                               true_labels, colors="red")

            # Save result
            plt.figure(figsize=(12, 12))
            plt.imshow(output_image.permute(1, 2, 0))
            plt.savefig(f'{save_dir}{idx}_{n}')
            plt.close()


def generate_metrics(stats, save_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    precisions = stats[0:4]
    recalls = stats[4:8]
    x = np.arange(4)
    width = 0.10

    # plot data in grouped manner of bar type
    plt.figure(figsize=(7, 12))
    plt.bar(x - 0.15, precisions, width, color='green')
    plt.bar(x, recalls, width, color='orange')
    plt.title('Test Set Metrics (IoU Threshold = 0.75)', fontsize=13)
    plt.xticks(x, ['All', 'Small', 'Medium', 'Large'])
    plt.xlabel("Tank Sizes")
    plt.ylabel("Score")
    plt.legend(["Precision", "Recall"])

    # add grid lines and change tick frequency on y-axis
    plt.grid(color='grey', linestyle=':', linewidth=1.0, axis='y', alpha=0.5)
    plt.yticks(np.arange(0, 1.04, 0.04))
    plt.axhline(color='red', linestyle='-', y=1.00)

    plt.savefig(f'{save_dir}accuracies_barplot.jpg')