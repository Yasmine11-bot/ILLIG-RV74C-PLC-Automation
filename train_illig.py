from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def main():
    # 1. Configuration du DataModule (sans argument invalide)
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

    # 2. Modèle Patchcore
    model = Patchcore(
        backbone="resnet18",
        coreset_sampling_ratio=0.1,
    )

    # 3. Engine d'exécution
    engine = Engine(
        default_root_dir=r"C:\AI_ILLIG\results_illig",
    )

    print("--- Lancement du Training ---")
    engine.fit(model=model, datamodule=datamodule)

    print("--- Lancement du Testing ---")
    engine.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()