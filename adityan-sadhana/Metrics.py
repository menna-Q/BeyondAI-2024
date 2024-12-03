import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

def evaluate_model(model, graphs, threshold=0.1):
    model.eval()
    with torch.no_grad():
        metrics_per_sample = []
        
        for i, graph in enumerate(graphs):
            predicted_positions = model(graph)
            true_positions = graph.x

            predicted_positions_np = predicted_positions.cpu().numpy()
            true_positions_np = true_positions.cpu().numpy()

            mse = mean_squared_error(true_positions_np, predicted_positions_np)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_positions_np, predicted_positions_np)
            
            is_correct = np.all(np.abs(predicted_positions_np - true_positions_np) <= threshold, axis=1)
            accuracy = np.mean(is_correct)
            f1 = f1_score(is_correct, is_correct)

            metrics_per_sample.append((mse, rmse, mae, accuracy, f1))
        
        metrics_per_sample_np = np.array(metrics_per_sample)
        avg_mse = np.mean(metrics_per_sample_np[:, 0])
        avg_rmse = np.mean(metrics_per_sample_np[:, 1])
        avg_mae = np.mean(metrics_per_sample_np[:, 2])
        avg_accuracy = np.mean(metrics_per_sample_np[:, 3])
        avg_f1 = np.mean(metrics_per_sample_np[:, 4])

        print("\nOverall Average Evaluation Metrics:")
        print(f"  Mean Squared Error (MSE): {avg_mse:.4f}")
        print(f"  Mean Absolute Error (MAE): {avg_mae:.4f}")
        print(f"  Accuracy: {avg_accuracy:.4%}")
        print(f"  F1 Score: {avg_f1:.4f}")
        print("=" * 50)

        plt.figure(figsize=(14, 6))
        sns.set_style('whitegrid')

        plt.subplot(1, 2, 1)
        plt.plot(metrics_per_sample_np[:, 0], label='MSE', color='blue')
        plt.plot(metrics_per_sample_np[:, 1], label='RMSE', color='green')
        plt.plot(metrics_per_sample_np[:, 2], label='MAE', color='orange')
        plt.plot(metrics_per_sample_np[:, 3], label='Accuracy', color='red')
        plt.plot(metrics_per_sample_np[:, 4], label='F1 Score', color='purple')
        plt.title('Model Performance Across Test Samples')
        plt.xlabel('Frames')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.tight_layout()
        plt.subplot(1, 2, 2)
        sns.histplot(metrics_per_sample_np[:, 3], color='red', kde=True, label='Accuracy')
        sns.histplot(metrics_per_sample_np[:, 4], color='purple', kde=True, label='F1 Score')
        plt.title('Distribution of Accuracy and F1 Score')
        plt.xlabel('Metric Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.suptitle('', fontsize=16)
        plt.show()

evaluate_model(model, test_graphs, threshold=0.1)
