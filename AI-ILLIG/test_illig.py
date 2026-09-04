from anomalib.data import PredictDataset
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ============================================================
# CHEMIN DU MODELE ENTRAINE
# ============================================================

MODEL_PATH = (
    r"C:\AI_ILLIG\results_illig\Patchcore"
    r"\illig_piece\v13\weights\lightning\model.ckpt"
)


# ============================================================
# IMAGE A TESTER
# ============================================================

IMAGE_PATH = r"C:\AI_ILLIG\illig_dataset\test\anomaly\anomalie_2.jpg"


# ============================================================
# PROGRAMME
# ============================================================

def main():

    print("======================================")
    print("      TEST DU MODELE PATCHCORE")
    print("======================================")

    print()
    print("Modèle :")
    print(MODEL_PATH)

    print()
    print("Image :")
    print(IMAGE_PATH)

    # --------------------------------------------------------
    # 1. Créer le même modèle PatchCore que pendant
    #    l'entraînement
    # --------------------------------------------------------

    model = Patchcore(
        backbone="resnet18",
        coreset_sampling_ratio=0.1,
    )

    # --------------------------------------------------------
    # 2. Créer l'Engine
    # --------------------------------------------------------

    engine = Engine()

    # --------------------------------------------------------
    # 3. Préparer l'image
    # --------------------------------------------------------

    dataset = PredictDataset(
        path=IMAGE_PATH,
        image_size=(256, 256),
    )

    # --------------------------------------------------------
    # 4. Faire la prédiction
    # --------------------------------------------------------

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=MODEL_PATH,
    )

    # --------------------------------------------------------
    # 5. Afficher le résultat
    # --------------------------------------------------------

    print()
    print("======================================")
    print("           RESULTAT")
    print("======================================")

    if predictions is None:
        print("Aucune prédiction retournée.")
        return

    for prediction in predictions:

        print()
        print("Image :", prediction.image_path)

        print("Score d'anomalie :",
              float(prediction.pred_score))

        print("Label :",
              int(prediction.pred_label))

        if int(prediction.pred_label) == 0:
            print()
            print("RESULTAT : OK")
        else:
            print()
            print("RESULTAT : NOK")

        print()
        print("======================================")


if __name__ == "__main__":
    main()
