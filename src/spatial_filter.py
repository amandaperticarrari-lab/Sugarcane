import numpy as np

def apply_spatial_morphological_filter(object_prediction_list, lambda_max=0.10, tile_size=1024):
    """
    Aplica o Filtro Morfológico Espacial (lambda_k <= lambda_max) nas predições do SAHI.
    Filtra falsos positivos causados pelo fechamento do dossel da cana-de-açúcar.
    """
    validated_predictions = []

    for pred in object_prediction_list:
        if pred.mask is not None:
            mask_area = np.sum(pred.mask.bool_mask)
            tile_area = tile_size * tile_size
            relative_area = mask_area / tile_area
            
            if relative_area <= lambda_max:
                validated_predictions.append(pred)

    return validated_predictions