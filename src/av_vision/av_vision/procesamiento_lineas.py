# ==========================================================
# MÓDULO DE PROCESAMIENTO DE LÍNEAS (COLOR + CANNY + HISTOGRAMA)
# Migrado de meraki_bringup → av_vision
# Ajustes: resolución 640x480, parámetros Hough recalibrados
# ==========================================================
import cv2
import numpy as np
 
# Parámetros Hough recalibrados para 640x480
HOUGH_THRESHOLD      = 60
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP   = 20
PENDIENTE_MIN_ABS    = 0.4
 
 
# ==========================================================
# DETECCIÓN DE LÍNEAS POR COLOR Y BORDES
# ==========================================================
def detectar_lineas_color_y_bordes(frame, zona):
    """
    Detecta líneas combinando máscara HSV y bordes Canny.
    Parámetro zona: 'Centro' (línea amarilla) | 'Izquierda'/'Derecha' (blancas)
 
    Retorna:
        resultados : list of (x1, y1, x2, y2, pendiente, longitud)
        combined_edges : imagen fusionada para debug
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    if zona == "Centro":  # Amarillo
        lower = np.array([18, 80,  80])
        upper = np.array([35, 255, 255])
    else:                 # Blanco
        lower = np.array([0,  0,   200])
        upper = np.array([180, 55, 255])
 
    mask_color = cv2.inRange(hsv, lower, upper)
 
    gris  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gris, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 150)
 
    combined_edges = cv2.addWeighted(edges, 0.5, mask_color, 0.8, 0)
 
    lineas = cv2.HoughLinesP(
        combined_edges,
        1, np.pi / 180,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )
 
    resultados = []
    if lineas is not None:
        for l in lineas:
            x1, y1, x2, y2 = l[0]
            if x2 == x1:
                continue
            pendiente = (y2 - y1) / (x2 - x1)
            if abs(pendiente) < PENDIENTE_MIN_ABS:
                continue
            longitud = np.hypot(x2 - x1, y2 - y1)
            resultados.append((x1, y1, x2, y2, pendiente, longitud))
 
    resultados.sort(key=lambda x: x[5], reverse=True)
    return resultados, combined_edges
 
 
# ==========================================================
# ANÁLISIS COMPLETO: IZQUIERDA + CENTRO + DERECHA
# ==========================================================
def detectar_y_analizar_lineas(frame):
    """
    Procesa las tres zonas del carril y selecciona la mejor línea por zona.
 
    Retorna:
        resultado : frame BGR con líneas dibujadas
        analisis  : dict por zona →
                      pendiente_prom  (float)
                      longitud_prom   (float)
                      intermitencia   (int)   número de candidatos detectados
                      x_base          (float) posición en píxeles en la base
    """
    # ── Resolución de trabajo: 640x480 (óptimo RPi5) ──────────────────────────
    frame = cv2.resize(frame, (640, 480))
    h, w  = frame.shape[:2]   # h=480, w=640
 
    # ── ROI trapezoidal general ───────────────────────────────────────────────
    trapezoide = np.zeros((h, w), dtype=np.uint8)
    roi_vertices = np.array([[
        (0,        h),
        (w,        h),
        (w // 2 + 270, int(h * 0.30)),
        (w // 2 - 270, int(h * 0.30))
    ]], dtype=np.int32)
    cv2.fillPoly(trapezoide, roi_vertices, 255)
 
    # ── Zona central (línea amarilla) ─────────────────────────────────────────
    centro_vertices = np.array([
        (int(w * 0.45), int(h * 0.35)),
        (int(w * 0.55), int(h * 0.35)),
        (int(w * 0.90), h),
        (int(w * 0.10), h)
    ], dtype=np.int32)
    mask_centro = np.zeros_like(trapezoide)
    cv2.fillPoly(mask_centro, [centro_vertices], 255)
    mask_centro = cv2.bitwise_and(mask_centro, trapezoide)
 
    resto    = cv2.subtract(trapezoide, mask_centro)
    mask_izq = resto.copy(); mask_izq[:, int(w / 2):] = 0
    mask_der = resto.copy(); mask_der[:, :int(w / 2)] = 0
 
    zonas = {
        "Izquierda": mask_izq,
        "Centro":    mask_centro,
        "Derecha":   mask_der
    }
    colores = {
        "Izquierda": (255,   0,   0),
        "Centro":    (  0,   0, 255),
        "Derecha":   (  0, 255,   0)
    }
 
    resultado = frame.copy()
    analisis  = {}
 
    margen_centro = w * 0.10
    margen_borde  = w * 0.05
 
    for nombre, mask in zonas.items():
        lineas_globales, _ = detectar_lineas_color_y_bordes(frame, nombre)
 
        lineas_filtradas = []
        for (x1, y1, x2, y2, m, L) in lineas_globales:
            # Signo de pendiente según zona
            if nombre == "Izquierda" and m >= 0:
                continue
            if nombre == "Derecha"   and m <= 0:
                continue
 
            xm, ym = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if not (0 <= xm < w and 0 <= ym < h):
                continue
            if mask[ym, xm] == 0:
                continue
 
            lineas_filtradas.append((x1, y1, x2, y2, m, L))
 
        pendiente_prom = 0.0
        long_prom      = 0.0
        intermitencia  = len(lineas_filtradas)
        x_base         = None
 
        if lineas_filtradas:
            x1, y1, x2, y2, m, L = max(lineas_filtradas, key=lambda x: x[5])
 
            if x2 != x1:
                m_ext = (y2 - y1) / (x2 - x1)
                b_ext = y1 - m_ext * x1
 
                y_bottom = h - 1
                y_top    = int(h * 0.30)
 
                x_bottom = int((y_bottom - b_ext) / m_ext)
                x_top    = int((y_top    - b_ext) / m_ext)
 
                x_bottom = max(0, min(w - 1, x_bottom))
                x_top    = max(0, min(w - 1, x_top))
 
                if y_bottom >= h * 0.8:
                    es_valida = True
                    if nombre == "Izquierda":
                        if not (margen_borde < x_bottom < w / 2 - margen_centro):
                            es_valida = False
                    elif nombre == "Centro":
                        if not (w / 2 - margen_centro < x_bottom < w / 2 + margen_centro):
                            es_valida = False
                    elif nombre == "Derecha":
                        if not (w / 2 + margen_centro < x_bottom < w - margen_borde):
                            es_valida = False
 
                    if es_valida:
                        L_ext = float(np.hypot(x_top - x_bottom, y_top - y_bottom))
                        cv2.line(resultado,
                                 (x_bottom, y_bottom),
                                 (x_top,    y_top),
                                 colores[nombre], 3)
                        pendiente_prom = float(m_ext)
                        long_prom      = L_ext
                        x_base         = float(x_bottom)
 
        if x_base is None:
            x_base = float(w / 2.0)
 
        analisis[nombre] = {
            "pendiente_prom": round(pendiente_prom, 3),
            "longitud_prom":  round(long_prom, 2),
            "intermitencia":  intermitencia,
            "x_base":         x_base
        }
 
    return resultado, analisis