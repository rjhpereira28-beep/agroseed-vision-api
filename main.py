from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import math

app = FastAPI(title="AgroSeed - Motor de Visión Artificial")

# Permite la conexión directa desde Lovable sin bloqueos de seguridad
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def inicio():
    return {"status": "ok", "mensaje": "API de AgroSeed Visión activa"}

@app.post("/analizar-tarjeta")
async def api_analizar_tarjeta(
    archivo: UploadFile = File(...),
    ancho_mm: float = Form(26.0),
    alto_mm: float = Form(76.0),
    factor_dispersion: float = Form(2.59),
    margen_clips_pct: float = Form(0.15)
):
    contenido = await archivo.read()
    nparr = np.frombuffer(contenido, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Formato de imagen inválido"}

    alto_px, ancho_px = img.shape[:2]

    # Exclusión de la zona superior e inferior tapada por clips de sujeción
    y_inicio = int(alto_px * margen_clips_pct)
    y_fin = int(alto_px * (1.0 - margen_clips_pct))
    zona_util = img[y_inicio:y_fin, :]

    alto_util_px = y_fin - y_inicio
    ancho_util_px = ancho_px

    # Cálculo dimensional real del área evaluada
    alto_util_mm = alto_real_mm = alto_mm * (1.0 - 2 * margen_clips_pct)
    ancho_util_mm = ancho_mm
    area_util_cm2 = (ancho_util_mm / 10.0) * (alto_util_mm / 10.0)

    # Conversión física: Micrones por píxel
    micrones_px = ((ancho_util_mm * 1000.0) / ancho_util_px + (alto_util_mm * 1000.0) / alto_util_px) / 2.0

    # Filtro HSV para aislar gotas azules sobre el papel amarillo
    hsv = cv2.cvtColor(zona_util, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, np.array([90, 40, 30]), np.array([160, 255, 255]))

    # Porcentaje de cobertura
    pixeles_azules = cv2.countNonZero(mascara)
    pct_cobertura = (pixeles_azules / (ancho_util_px * alto_util_px)) * 100.0

    # Conteo y medición de manchas individuales
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diametros_reales = []
    volumenes_gotas = []

    for c in contornos:
        area_px = cv2.contourArea(c)
        if area_px < 2.0:
            continue
        d_mancha_um = 2.0 * math.sqrt(area_px / math.pi) * micrones_px
        d_gota_um = d_mancha_um / factor_dispersion
        diametros_reales.append(d_gota_um)

        d_cm = d_gota_um / 10000.0
        volumenes_gotas.append((math.pi / 6.0) * (d_cm ** 3))

    total_gotas = len(diametros_reales)
    if total_gotas == 0:
        return {
            "cobertura_gotas_cm2": 0.0,
            "porcentaje_cobertura": 0.0,
            "dm_vmd": 0.0,
            "dmn": 0.0,
            "dv01": 0.0,
            "dv09": 0.0,
            "amplitud_relativa": 0.0,
            "desvio": 0.0,
            "volumen_teorico_lt_ha": 0.0
        }

    # Distribución volumétrica acumulada
    indices = np.argsort(diametros_reales)
    d_sorted = np.array(diametros_reales)[indices]
    v_sorted = np.array(volumenes_gotas)[indices]

    vol_total = np.sum(v_sorted)
    vol_acum = np.cumsum(v_sorted) / vol_total

    dv01 = float(np.interp(0.10, vol_acum, d_sorted))
    dv05 = float(np.interp(0.50, vol_acum, d_sorted))
    dv09 = float(np.interp(0.90, vol_acum, d_sorted))

    return {
        "cobertura_gotas_cm2": round(total_gotas / area_util_cm2, 1),
        "porcentaje_cobertura": round(pct_cobertura, 1),
        "dm_vmd": round(dv05, 0),
        "dmn": round(float(np.median(diametros_reales)), 0),
        "dv01": round(dv01, 0),
        "dv09": round(dv09, 0),
        "amplitud_relativa": round((dv09 - dv01) / dv05 if dv05 > 0 else 0.0, 2),
        "desvio": round(float(np.std(diametros_reales)), 1),
        "volumen_teorico_lt_ha": round((vol_total / area_util_cm2) * 100000.0, 1)
    }
