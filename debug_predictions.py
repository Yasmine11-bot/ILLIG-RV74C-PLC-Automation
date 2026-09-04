import torch
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def debug():
    datamodule = Folder(
        name="illig_piece",
        root=r"C:\AI_ILLIG\illig_dataset",
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/anomaly",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )
    
    model = Patchcore(backbone="resnet18")
    engine = Engine(default_root_dir=r"C:\AI_ILLIG\results_illig")
    
    engine.fit(model=model, datamodule=datamodule)
    
    # Prédiction sur le dataset de test
    predictions = engine.predict(model=model, datamodule=datamodule)
    
    print("\n" + "="*60)
    print("DÉTAIL DES PRÉDICTIONS IMAGE PAR IMAGE")
    print("="*60)
    
    for batch in predictions:
        path = batch.image_path[0]
        
        # Récupération sécurisée du vrai label (gt_label)
        gt = batch.gt_label.item() if hasattr(batch, "gt_label") and batch.gt_label is not None else None
        label_reel = "ANOMALIE (1)" if gt == 1 else "NORMAL (0)" if gt == 0 else "INCONNU"
        
        # Récupération du score d'anomalie
        score = batch.pred_score.item() if hasattr(batch, "pred_score") else 0.0
        
        # Récupération de la prédiction du modèle
        pred_label = batch.pred_label.item() if hasattr(batch, "pred_label") else None
        pred = "ANOMALIE (1)" if pred_label == 1 else "NORMAL (0)"
        
        print(f"Fichier    : {path}")
        print(f"Label Réel : {label_reel}")
        print(f"Score Anom : {score:.4f} -> Prédit : {pred}")
        print("-" * 60)

if __name__ == "__main__":
    debug()