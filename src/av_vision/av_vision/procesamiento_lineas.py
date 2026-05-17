# ==========================================================
# MÓDULO DE PROCESAMIENTO DE LÍNEAS (COLOR + CANNY)
# Migrado de meraki_bringup → av_vision
# Ajustes: resolución 640x480, parámetros Hough recalibrados
# Mejoras: CLAHE, Canny dinámico, polígonos calibrados,
#          filtro color+signo, top-3 ponderado, EMA, confianza
#
# Compatibilidad con vision_node.py:
#   analisis['Izquierda'] → línea amarilla (carril central colombiano)
#   analisis['Centro']    → estimación geométrica entre ambas líneas
#   analisis['Derecha']   → línea blanca (borde derecho del carril)
# ==========================================================
import cv2
import numpy as np
import json
import os

# Parámetros Hough
HOUGH_THRESHOLD       = 25
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP    = 100
PENDIENTE_MIN_ABS     = 0.3
PENDIENTE_MAX_ABS     = 0.95

# Suavizado temporal
EMA_ALPHA  = 0.25
_ema_state = {
    "Izquierda": {"x_base": None, "pendiente": None},
    "Derecha":   {"x_base": None, "pendiente": None},
}

# CLAHE
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Polígonos calibrados
POLY_FILE = "poligonos.json"
_polys    = None


# ==========================================================
# CARGA DE POLÍGONOS
# Llamar una vez en el __init__ del nodo ROS:
#   from av_vision.procesamiento_lineas import cargar_poligonos
#   cargar_poligonos()
# ==========================================================
def cargar_poligonos(w=640, h=480):
    global _polys
    if os.path.exists(POLY_FILE):
        with open(POLY_FILE) as f:
            data = json.load(f)
        # El calibrador guarda 'centro' y 'derecha'
        # Los mapeamos a 'Izquierda' y 'Derecha' para el nodo ROS
        _polys = {
            "Izquierda": np.array(data["centro"],  dtype=np.int32),
            "Derecha":   np.array(data["derecha"], dtype=np.int32),
        }
    else:
        # Polígonos por defecto — generar con calibrador offline
        _polys = {
            "Izquierda": np.array([
                [int(w * 0.05), h],
                [int(w * 0.45), h],
                [int(w * 0.35), int(h * 0.45)],
                [int(w * 0.10), int(h * 0.45)],
            ], dtype=np.int32),
            "Derecha": np.array([
                [int(w * 0.45), h],
                [int(w * 0.95), h],
                [int(w * 0.85), int(h * 0.45)],
                [int(w * 0.55), int(h * 0.45)],
            ], dtype=np.int32),
        }
    return _polys


# ==========================================================
# DETECCIÓN DE LÍNEAS POR COLOR Y BORDES
# ==========================================================
def detectar_lineas_color_y_bordes(frame, zona):
    """
    Detecta líneas combinando máscara HSV + Canny dinámico + morfológico.

    zona: 'Izquierda' (línea amarilla) | 'Derecha' (línea blanca)

    Retorna:
        resultados     : list of (x1, y1, x2, y2, pendiente, longitud)
        combined_edges : imagen fusionada para debug
    """
    global _polys

    # CLAHE sobre canal V
    hsv_eq = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_eq[:, :, 2] = _clahe.apply(hsv_eq[:, :, 2])
    frame_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    hsv = hsv_eq

    if zona == "Izquierda":   # Amarillo — línea central colombiana
        lower = np.array([15,  40,  80])
        upper = np.array([40, 255, 255])
    else:                     # Blanco — línea derecha
        lower = np.array([0,   0,  210])
        upper = np.array([180, 35, 255])

    mask_color = cv2.inRange(hsv, lower, upper)

    # Canny dinámico
    gris    = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2GRAY)
    mediana = float(np.median(gris))
    sigma   = 0.33
    blur    = cv2.GaussianBlur(gris, (5, 5), 0)
    edges   = cv2.Canny(blur,
                        int(max(0,   (1.0 - sigma) * mediana)),
                        int(min(255, (1.0 + sigma) * mediana)))

    combined_edges = cv2.addWeighted(edges, 0.5, mask_color, 0.8, 0)

    # Morfológico
    kernel         = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN,  kernel, iterations=1)
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Aplica polígono
    if _polys is not None and zona in _polys:
        mask_poly = np.zeros_like(combined_edges)
        cv2.fillPoly(mask_poly, [_polys[zona]], 255)
        combined_edges = cv2.bitwise_and(combined_edges, mask_poly)

    lineas = cv2.HoughLinesP(
        combined_edges,
        1, np.pi / 180,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    resultados = []
    if lineas is not None:
        for l in lineas:
            x1, y1, x2, y2 = l[0]
            if x2 == x1:
                continue
            pendiente = (y2 - y1) / (x2 - x1)
            if not (PENDIENTE_MIN_ABS < abs(pendiente) < PENDIENTE_MAX_ABS):
                continue
            # Filtro de signo según perspectiva real de la cámara:
            # Izquierda (amarilla) → pendiente negativa
            # Derecha   (blanca)   → pendiente positiva
            if zona == "Izquierda" and pendiente > 0:
                continue
            if zona == "Derecha"   and pendiente < 0:
                continue
            longitud = float(np.hypot(x2 - x1, y2 - y1))
            resultados.append((x1, y1, x2, y2, pendiente, longitud))

    resultados.sort(key=lambda x: x[5], reverse=True)
    return resultados, combined_edges


# ==========================================================
# ANÁLISIS COMPLETO — devuelve Izquierda + Centro + Derecha
# Compatible con vision_node.py sin modificarlo
# ==========================================================
def detectar_y_analizar_lineas(frame):
    """
    Procesa las dos líneas del carril y devuelve el análisis en el formato
    que espera vision_node.py (Izquierda, Centro, Derecha).

    Retorna:
        resultado : frame BGR 640x480 con líneas y área del carril dibujadas
        analisis  : {
            'Izquierda': {pendiente_prom, longitud_prom, intermitencia, x_base, confianza},
            'Centro':    {pendiente_prom, longitud_prom, intermitencia, x_base, confianza},
            'Derecha':   {pendiente_prom, longitud_prom, intermitencia, x_base, confianza},
          }
    """
    global _polys

    frame = cv2.resize(frame, (640, 480))
    h, w  = frame.shape[:2]

    if _polys is None:
        cargar_poligonos(w, h)

    colores = {
        "Izquierda": (0,   200, 255),   # amarillo-cyan
        "Derecha":   (255, 255, 255),   # blanco
    }

    resultado = frame.copy()
    analisis  = {}
    extremos  = {}

    for nombre in ["Izquierda", "Derecha"]:
        lineas_globales, _ = detectar_lineas_color_y_bordes(frame, nombre)

        pendiente_prom = 0.0
        long_prom      = 0.0
        intermitencia  = len(lineas_globales)
        x_base         = None

        if lineas_globales:
            # Promedio ponderado top-3
            top3    = lineas_globales[:3]
            total_L = sum(s[5] for s in top3)
            m_ext   = sum(s[4] * s[5] for s in top3) / total_L
            x1r, y1r = top3[0][0], top3[0][1]
            b_ext   = y1r - m_ext * x1r

            if abs(m_ext) > 1e-6 and _polys is not None:
                poly     = _polys[nombre]
                y_bottom = int(np.max(poly[:, 1]))
                y_top    = int(np.min(poly[:, 1]))

                x_bottom = max(0, min(w - 1, int((y_bottom - b_ext) / m_ext)))
                x_top    = max(0, min(w - 1, int((y_top    - b_ext) / m_ext)))

                L_ext = float(np.hypot(x_top - x_bottom, y_top - y_bottom))

                cv2.line(resultado,
                         (x_bottom, y_bottom),
                         (x_top,    y_top),
                         colores[nombre], 3, cv2.LINE_AA)

                pendiente_prom = float(m_ext)
                long_prom      = L_ext
                x_base         = float(x_bottom)
                extremos[nombre] = ((x_bottom, y_bottom), (x_top, y_top))

        # EMA
        s = _ema_state[nombre]
        if x_base is not None:
            if s["x_base"] is None:
                s["x_base"]    = x_base
                s["pendiente"] = pendiente_prom
            else:
                s["x_base"]    = EMA_ALPHA * x_base        + (1 - EMA_ALPHA) * s["x_base"]
                s["pendiente"] = EMA_ALPHA * pendiente_prom + (1 - EMA_ALPHA) * s["pendiente"]
            x_base         = s["x_base"]
            pendiente_prom = s["pendiente"]
        else:
            x_base = s["x_base"] if s["x_base"] is not None else float(w / 2.0)

        # Confianza
        score_n   = min(1.0, intermitencia / 5.0)
        score_l   = min(1.0, long_prom / 150.0)
        confianza = round(0.6 * score_n + 0.4 * score_l, 3)

        analisis[nombre] = {
            "pendiente_prom": round(pendiente_prom, 3),
            "longitud_prom":  round(long_prom, 2),
            "intermitencia":  intermitencia,
            "x_base":         round(x_base, 2),
            "confianza":      confianza,
        }

    # Centro geométrico entre las dos líneas
    x_izq = analisis["Izquierda"]["x_base"]
    x_der = analisis["Derecha"]["x_base"]
    m_izq = analisis["Izquierda"]["pendiente_prom"]
    m_der = analisis["Derecha"]["pendiente_prom"]
    inter_cen = analisis["Izquierda"]["intermitencia"] + analisis["Derecha"]["intermitencia"]

    analisis["Centro"] = {
        "pendiente_prom": round((m_izq + m_der) / 2.0, 3),
        "longitud_prom":  0.0,
        "intermitencia":  inter_cen,
        "x_base":         round((x_izq + x_der) / 2.0, 2),
        "confianza":      round(min(analisis["Izquierda"]["confianza"],
                                   analisis["Derecha"]["confianza"]), 3),
    }

    # Área del carril
    if "Izquierda" in extremos and "Derecha" in extremos:
        ei = extremos["Izquierda"]
        ed = extremos["Derecha"]
        poligono = np.array([ei[0], ei[1], ed[1], ed[0]], dtype=np.int32)
        overlay  = resultado.copy()
        cv2.fillPoly(overlay, [poligono], (0, 200, 100))
        cv2.addWeighted(overlay, 0.20, resultado, 0.80, 0, resultado)

    return resultado, analisis