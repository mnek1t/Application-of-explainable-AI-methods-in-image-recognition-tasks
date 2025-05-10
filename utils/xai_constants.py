import quantus
import numpy as np

SPLIT_DATASET_PATH = "D:\\University\\Bachelor Thesis\\garbadge_dataset\\splitted_augmented_dataset"
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\results\\best_inception_model.keras"
IMAGE_PATH = "D:\\University\\Bachelor Thesis\\xai methods\\bcaa-trash.jpg"
TEST_DATASET_PATH = "D:\\University\\Bachelor Thesis\\garbadge_dataset\\splitted_augmented_dataset\\test"
TARGET_SIZE = (224, 224)
INTENSITY = 0.5
METRICS = {
    "Robustness": quantus.AvgSensitivity(
        nr_samples=2,
        lower_bound=0.2,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference,
        abs=True,
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True,
        display_progressbar=True
    ),
    "Faithfulness": quantus.FaithfulnessCorrelation(
        nr_runs=15,
        subset_size=20,
        perturb_baseline="black",
        perturb_func= quantus.baseline_replacement_by_indices,
        similarity_func= quantus.similarity_func.correlation_pearson,
        return_aggregate=True,
        display_progressbar=True
    ),
    "Complexity":  quantus.Sparseness(
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True
    ),
    "Effective Complexity":  quantus.EffectiveComplexity(
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True,
    ),
    "Localisation": quantus.RelevanceRankAccuracy(
        abs=True,
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True,
    ),
    "Selectivity": quantus.Selectivity(
        perturb_baseline="black",
        patch_size=16,
        perturb_func=quantus.baseline_replacement_by_indices,
        abs=True,
        normalise=True,
        return_aggregate=True,
        display_progressbar=True
    ),
    "SensitivityN": quantus.SensitivityN(
        features_in_step=256, 
        n_max_percentage=0.3, 
        similarity_func=lambda a, b, **kwargs: quantus.similarity_func.abs_difference(np.array(a), np.array(b)),
        perturb_baseline="black",
        perturb_func=quantus.baseline_replacement_by_indices,
        return_aggregate=True,
        display_progressbar=True
    )
}