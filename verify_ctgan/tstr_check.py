import sys
sys.path.insert(0, '/home/lbrum14/projects/Katabatic')

from katabatic.evaluate.tstr.evaluation import TSTREvaluation

for i in range(1, 4):
    print(f"=== Run {i} ===")
    e = TSTREvaluation(
        synthetic_dir="verify_ctgan/pipeline",
        real_test_dir="sample_data/adult",
    )
    results = e.evaluate()
    for model, metrics in results.items():
        auc = metrics.get("AUC", float("nan"))
        print(f"  {model}: acc={metrics['Accuracy']:.4f}  f1={metrics['F1 Score']:.4f}  auc={auc:.4f}")
    print()
