import os
import sys
from pathlib import Path
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from spatial_filter import apply_spatial_morphological_filter

def find_test_image(base_dir="/home/aperticarrari/yolo_sugarcane"):
    """Busca automaticamente por uma imagem válida dentro do diretório do projeto."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    for ext in extensions:
        matches = list(Path(base_dir).rglob(ext))
        if matches:
            selected_img = str(matches[0])
            print(f"-> Imagem de teste localizada automaticamente: {selected_img}")
            return selected_img
    
    print(f"\n[ERRO CRÍTICO] Nenhuma imagem (.jpg, .png, .tif) foi encontrada em {base_dir}")
    sys.exit(1)

def run_sugarcane_pipeline(
    image_path="/home/aperticarrari/yolo_sugarcane/image/test_image.jpg",
    weights_path="/home/aperticarrari/yolo_sugarcane/runs/segment/runs/sugarcane_weed_512x512_artigo_10/weights/best.pt",
    lambda_max=0.10,
    conf_threshold=0.25
):
    if not os.path.exists(weights_path):
        print(f"\n[ERRO] O arquivo de pesos não foi encontrado em:\n{weights_path}")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"[AVISO] Imagem padrão não encontrada em: {image_path}")
        print("Procurando imagem de teste no projeto...")
        image_path = find_test_image()

    print(f"Carregando modelo: {weights_path}")
    
    # 1. Carrega o modelo YOLOv8-seg via SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=weights_path,
        confidence_threshold=conf_threshold,
        device="cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    )

    # 2. Executa a inferência fatiada SAHI (1024x1024 px, 20% overlap)
    print("Executando fatiamento SAHI no ortomosaico...")
    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.20,
        overlap_width_ratio=0.20
    )

    # 3. Filtro Morfológico Espacial importado do módulo modularizado
    validated_predictions = apply_spatial_morphological_filter(
        object_prediction_list=result.object_prediction_list,
        lambda_max=lambda_max,
        tile_size=1024
    )

    print("\n--- Resultados do Pipeline ---")
    print(f"Detecções brutas (Raw Masks M_k): {len(result.object_prediction_list)}")
    print(f"Detecções validadas pelo Filtro Morfológico (Validated Masks M_k^hat): {len(validated_predictions)}")

    return validated_predictions

if __name__ == "__main__":
    run_sugarcane_pipeline()