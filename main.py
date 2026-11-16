from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import cv2
import numpy as np
import math

app = FastAPI(title="AgroSeed - Motor de Visión Artificial y Calibración")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PuntoFaja(BaseModel):
    posicion_m: float
    volumen_lt_ha: float

class SolicitudSimulacion(BaseModel):
    puntos: List[PuntoFaja]
    ancho_min_m: float = 14.0
    ancho_max_m: float = 24.0
    paso_m: float = 1.0

def clasificar_gota_asabe(dmv_um: float) -> str:
    if dmv_um <= 0:
        return "Sin detección"
    elif dmv_um < 60:
        return "Extremadamente Fina"
    elif dmv_um < 106:
        return "Muy Fina"
    elif dmv_um < 235:
        return "Fina"
    elif dmv_um < 340:
        return "Media"
    elif dmv_um < 403:
        return "Gruesa"
    elif dmv_um < 502:
        return "Muy Gruesa"
    elif dmv_um < 665:
        return "Extremadamente Gruesa"
    else:
        return "Ultra Gruesa"

def recortar_tarjeta_amarilla(img_bgr):
    """
    Detecta el rectángulo de la tarjeta hidrosensible (amarillo + manchas azules)
    y recorta el fondo oscuro/mesa exterior para evitar falsos positivos.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Máscara del papel amarillo (incluyendo tonos verdosos por luz)
    mask_amarillo = cv2.inRange(hsv, np.array([15, 50, 60]), np.array([45, 255, 255]))
    # Incluimos las manchas azules internas para que el contorno de la tarjeta sea sólido
    mask_azul = cv2.inRange(hsv, np.array([85, 40, 30]), np.array([165, 255, 255]))
    mask_tarjeta = cv2.bitwise_or(mask_amarillo, mask_azul)

    kernel = np.ones((9, 9), np.uint8)
    mask_tarjeta = cv2.morphologyEx(mask_tarjeta, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mask_tarjeta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return img_bgr

    mayor = max(contornos, key=cv2.contourArea)
    area_img = img_bgr.shape[0] * img_bgr.shape[1]

    # Si el área amarilla detectada ocupa al menos el 5% de la foto, recortamos a su caja con un margen interior del 4%
    if cv2.contourArea(mayor) > area_img * 0.05:
        x, y, w, h = cv2.boundingRect(mayor)
        margen_x = int(w * 0.04)
        margen_y = int(h * 0.04)
        x1 = max(0, x + margen_x)
        y1 = max(0, y + margen_y)
        x2 = min(img_bgr.shape[1], x + w - margen_x)
        y2 = min(img_bgr.shape[0], y + h - margen_y)
        if x2 > x1 and y2 > y1:
            return img_bgr[y1:y2, x1:x2]

    return img_bgr

@app.get("/")
def inicio():
    return {"status": "ok", "mensaje": "API de AgroSeed Visión y Calibración activa"}

@app.post("/analizar-tarjeta")
async def api_analizar_tarjeta(
    archivo: UploadFile = File(...),
    ancho_mm: float = Form(26.0),
    alto_mm: float = Form(76.0),
    factor_dispersion: float = Form(1.2),
    margen_clips_pct: float = Form(0.05)
):
    contenido = await archivo.read()
    nparr = np.frombuffer(contenido, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Formato de imagen inválido"}

    # 1. AISLAR EXCLUSIVAMENTE LA TARJETA AMARILLA (ignorar mesa o fondo)
    tarjeta = recortar_tarjeta_amarilla(img)
    alto_px, ancho_px = tarjeta.shape[:2]

    # Exclusión de borde superior/inferior donde va el clip
    y_inicio = int(alto_px * margen_clips_pct)
    y_fin = int(alto_px * (1.0 - margen_clips_pct))
    zona_util = tarjeta[y_inicio:y_fin, :]

    alto_util_px = y_fin - y_inicio
    ancho_util_px = ancho_px

    alto_util_mm = alto_mm * (1.0 - 2 * margen_clips_pct)
    ancho_util_mm = ancho_mm
    area_util_cm2 = (ancho_util_mm / 10.0) * (alto_util_mm / 10.0)

    micrones_px = ((ancho_util_mm * 1000.0) / ancho_util_px + (alto_util_mm * 1000.0) / alto_util_px) / 2.0

    # 2. DETECCIÓN PRECISA DE MANCHAS AZULES/ÍNDIGO SOBRE FONDO AMARILLO
    hsv = cv2.cvtColor(zona_util, cv2.COLOR_BGR2HSV)
    # Rango específico de tintes azules, cianes y morados propios de tarjeta hidrosensible
    mascara_azul = cv2.inRange(hsv, np.array([82, 45, 25]), np.array([165, 255, 255]))

    # Limpieza de ruido de 1 píxel
    kernel = np.ones((2, 2), np.uint8)
    mascara = cv2.morphologyEx(mascara_azul, cv2.MORPH_OPEN, kernel)

    pixeles_gotas = cv2.countNonZero(mascara)
    pct_cobertura = (pixeles_gotas / (ancho_util_px * alto_util_px)) * 100.0

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diametros_reales = []
    volumenes_gotas = []

    for c in contornos:
        area_px = cv2.contourArea(c)
        if area_px < 2.0:
            continue
        d_mancha_um = 2.0 * math.sqrt(area_px / math.pi) * micrones_px
        d_gota_um = d_mancha_um / factor_dispersion
        # Filtrar anomalías físicas mayores a 1,800 micrones
        if d_gota_um > 1800.0:
            continue
        diametros_reales.append(d_gota_um)

        d_cm = d_gota_um / 10000.0
        volumenes_gotas.append((math.pi / 6.0) * (d_cm ** 3))

    total_gotas = len(diametros_reales)
    if total_gotas == 0:
        return {
            "total_gotas": 0,
            "cobertura_gotas_cm2": 0.0,
            "porcentaje_cobertura": 0.0,
            "dm_vmd": 0.0,
            "dmn": 0.0,
            "dv01": 0.0,
            "dv09": 0.0,
            "amplitud_relativa": 0.0,
            "desvio": 0.0,
            "volumen_teorico_lt_ha": 0.0,
            "categoria_asabe": "Sin detección",
            "pct_gotas_derivables": 0.0,
            "pct_volumen_derivable": 0.0
        }

    d_arr = np.array(diametros_reales)
    v_arr = np.array(volumenes_gotas)

    indices = np.argsort(d_arr)
    d_sorted = d_arr[indices]
    v_sorted = v_arr[indices]

    vol_total = float(np.sum(v_sorted))
    vol_acum = np.cumsum(v_sorted) / vol_total

    dv01 = float(np.interp(0.10, vol_acum, d_sorted))
    dv05 = float(np.interp(0.50, vol_acum, d_sorted))
    dv09 = float(np.interp(0.90, vol_acum, d_sorted))

    mascara_deriva = d_arr < 150.0
    pct_gotas_derivables = (float(np.sum(mascara_deriva)) / total_gotas) * 100.0
    pct_volumen_derivable = (float(np.sum(v_arr[mascara_deriva])) / vol_total) * 100.0 if vol_total > 0 else 0.0

    return {
        "total_gotas": total_gotas,
        "cobertura_gotas_cm2": round(total_gotas / area_util_cm2, 1),
        "porcentaje_cobertura": round(pct_cobertura, 1),
        "dm_vmd": round(dv05, 0),
        "dmn": round(float(np.median(d_arr)), 0),
        "dv01": round(dv01, 0),
        "dv09": round(dv09, 0),
        "amplitud_relativa": round((dv09 - dv01) / dv05 if dv05 > 0 else 0.0, 2),
        "desvio": round(float(np.std(d_arr)), 1),
        "volumen_teorico_lt_ha": round((vol_total / area_util_cm2) * 100000.0, 1),
        "categoria_asabe": clasificar_gota_asabe(dv05),
        "pct_gotas_derivables": round(pct_gotas_derivables, 1),
        "pct_volumen_derivable": round(pct_volumen_derivable, 1)
    }

@app.post("/simular-faja")
async def api_simular_faja(datos: SolicitudSimulacion):
    if len(datos.puntos) < 2:
        return {"error": "Se requieren al menos 2 tarjetas con posición para simular el traslape."}

    puntos_ord = sorted(datos.puntos, key=lambda p: p.posicion_m)
    pos_x = np.array([p.posicion_m for p in puntos_ord], dtype=float)
    vol_y = np.array([p.volumen_lt_ha for p in puntos_ord], dtype=float)

    paso_malla = 0.25
    x_min, x_max = float(np.min(pos_x)), float(np.max(pos_x))
    x_perfil = np.arange(x_min, x_max + paso_malla, paso_malla)
    y_perfil = np.interp(x_perfil, pos_x, vol_y, left=0.0, right=0.0)

    def obtener_deposito(x_eval, desplazamiento, invertir=False):
        x_rel = x_eval - desplazamiento
        if invertir:
            x_rel = -x_rel
        return np.interp(x_rel, x_perfil, y_perfil, left=0.0, right=0.0)

    tabla_cv = []
    anchos_evaluar = np.arange(datos.ancho_min_m, datos.ancho_max_m + datos.paso_m, datos.paso_m)

    menor_cv_carrusel = 999.0
    curvas_optimas = {}

    for ancho in anchos_evaluar:
        ancho_f = float(ancho)
        x_eval = np.arange(-ancho_f / 2.0, (ancho_f / 2.0) + paso_malla, paso_malla)

        p_izq_rt = obtener_deposito(x_eval, -ancho_f, invertir=False)
        p_cen_rt = obtener_deposito(x_eval, 0.0, invertir=False)
        p_der_rt = obtener_deposito(x_eval, ancho_f, invertir=False)
        suma_rt = p_izq_rt + p_cen_rt + p_der_rt

        media_rt = float(np.mean(suma_rt))
        std_rt = float(np.std(suma_rt))
        cv_rt = (std_rt / media_rt * 100.0) if media_rt > 0 else 0.0

        p_izq_b2b = obtener_deposito(x_eval, -ancho_f, invertir=True)
        p_cen_b2b = obtener_deposito(x_eval, 0.0, invertir=False)
        p_der_b2b = obtener_deposito(x_eval, ancho_f, invertir=True)
        suma_b2b = p_izq_b2b + p_cen_b2b + p_der_b2b

        media_b2b = float(np.mean(suma_b2b))
        std_b2b = float(np.std(suma_b2b))
        cv_b2b = (std_b2b / media_b2b * 100.0) if media_b2b > 0 else 0.0

        tabla_cv.append({
            "ancho_faja_m": round(ancho_f, 1),
            "cv_carrusel_pct": round(cv_rt, 1),
            "cv_ida_vuelta_pct": round(cv_b2b, 1),
            "tasa_media_solapada_lt_ha": round(media_rt, 1)
        })

        if cv_rt < menor_cv_carrusel:
            menor_cv_carrusel = cv_rt
            x_graf = np.arange(-ancho_f * 1.5, (ancho_f * 1.5) + 1.0, 1.0)
            curvas_optimas = {
                "ancho_optimo_m": round(ancho_f, 1),
                "cv_optimo_carrusel_pct": round(cv_rt, 1),
                "tasa_media_lt_ha": round(media_rt, 1),
                "posiciones_m": [round(float(x), 1) for x in x_graf],
                "pasada_izquierda_lt_ha": [round(float(v), 2) for v in obtener_deposito(x_graf, -ancho_f, False)],
                "pasada_central_lt_ha": [round(float(v), 2) for v in obtener_deposito(x_graf, 0.0, False)],
                "pasada_derecha_lt_ha": [round(float(v), 2) for v in obtener_deposito(x_graf, ancho_f, False)],
                "total_carrusel_lt_ha": [
                    round(float(v), 2) for v in (
                        obtener_deposito(x_graf, -ancho_f, False) +
                        obtener_deposito(x_graf, 0.0, False) +
                        obtener_deposito(x_graf, ancho_f, False)
                    )
                ],
                "total_ida_vuelta_lt_ha": [
                    round(float(v), 2) for v in (
                        obtener_deposito(x_graf, -ancho_f, True) +
                        obtener_deposito(x_graf, 0.0, False) +
                        obtener_deposito(x_graf, ancho_f, True)
                    )
                ]
            }

    return {
        "tabla_uniformidad_cv": tabla_cv,
        "simulacion_grafica": curvas_optimas
    }
