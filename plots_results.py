import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_CSV = 'output/classification_results/all_results.csv'
RESULTS_FOLDER = "plots"

df = pd.read_csv(OUTPUT_CSV)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

metrics = ['cv_accuracy', 'test_accuracy', 'cv_tpr', 'test_tpr']
colors = ['blue', 'lightblue', 'green', 'lightgreen']

bar_width = 0.2
index = range(len(df['model']))

for i, (metric, color) in enumerate(zip(metrics, colors)):
    plt.bar([x + i*bar_width for x in index], df[metric],
            width=bar_width, label=metric, color=color, alpha=0.7)

plt.xlabel('Модели')
plt.ylabel('Метрики на перформанс')
plt.title('Споредба на модели')
plt.xticks([x + bar_width*1.5 for x in index], df['model'], rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, 'model_performance_comparison.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(df['model'], df['training_time'], color='red', alpha=0.7)
plt.xlabel('Модели')
plt.ylabel('Време (секунди)')
plt.title('Време на тренирање на моделите')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, 'model_training_time.png'))
plt.close()
