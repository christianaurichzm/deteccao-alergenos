from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(matrix, algorithm):
    class_labels = ['Alérgeno', 'Não alergeno']
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_labels)
    disp.plot(cmap='Purples', values_format='.2f')

    ax = plt.gca()
    ax.set_yticklabels(class_labels, rotation=90, va='center')

    plt.xlabel('Rótulo predito')
    plt.ylabel('Rótulo verdadeiro')
    plt.title(f'Matriz de confusão para o algoritmo: {algorithm.value}')

    plt.tight_layout()
    plt.show()


def plot_metrics(avg_accuracy, avg_precisions, avg_recalls, avg_f1, algorithm):
    metrics = ['Acurácia', 'Precisão', 'Revocação', 'Score F1']
    values = [avg_accuracy, avg_precisions, avg_recalls, avg_f1]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#003f5c', '#7a5195', '#ef5675', '#ffa600'])

    plt.ylabel('Pontuação')
    plt.title(f'Métricas para o algoritmo: {algorithm}')
    plt.ylim(0, 1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), ha='center', va='bottom')

    plt.show()
