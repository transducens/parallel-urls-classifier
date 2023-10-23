import sys
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Inicializa listas para las etiquetas reales y predichas
y_true = []
y_pred = []

# Leer las etiquetas desde sys.stdin
for line in sys.stdin:
    real, predicted = line.strip().split('\t')
    y_true.append(real)
    y_pred.append(predicted)

# Crear la matriz de confusión
confusion = confusion_matrix(y_true, y_pred)

# Calcula la normalización por fila para reflejar la proporción dentro de cada clase
confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

# Configurar el tamaño del gráfico en función del número de clases
num_classes = len(np.unique(y_true))
fig_size = max(12, num_classes // 5)  # Aumentar el tamaño del gráfico
plt.figure(figsize=(fig_size, fig_size))

# Generar el gráfico de la matriz de confusión con la normalización por fila
plt.imshow(confusion_normalized, interpolation='nearest', cmap=plt.get_cmap('Greys'))
plt.title('Matriz de Confusión (Normalizada por Clase)')
#plt.colorbar()

plt.axis('off')

#tick_marks = np.arange(num_classes)
#plt.xticks(tick_marks, rotation=90)
#plt.yticks(tick_marks)

# Etiquetas de las clases en los ejes
#plt.xticks(tick_marks, np.unique(y_true), fontsize=8)
#plt.yticks(tick_marks, np.unique(y_true), fontsize=8)

# Añadir los valores normalizados de la matriz en las celdas con espaciado adicional
#for i in range(num_classes):
#    for j in range(num_classes):
#        plt.text(j, i, f"{confusion_normalized[i, j]:.2f}", horizontalalignment="center", fontsize=4, color="white", va='center')

# Ajustar el espaciado entre las celdas de la matriz
plt.tight_layout()

# Guardar el gráfico como una imagen en el archivo especificado
output_filename = sys.argv[1]  # El nombre del archivo se proporciona como primer argumento
plt.savefig(output_filename, dpi=300)
