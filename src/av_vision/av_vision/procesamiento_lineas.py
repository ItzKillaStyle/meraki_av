import cv2
import numpy as np

# ==========================================================
# PERSPECTIVA CALIBRADA
# ==========================================================

SRC = np.float32([
    [78, 476],
    [578, 478],
    [386, 200],
    [259, 199]
])

DST = np.float32([
    [160, 480],
    [480, 480],
    [480, 0],
    [160, 0]
])

W = 640
H = 480

M  = cv2.getPerspectiveTransform(SRC, DST)
Mi = cv2.getPerspectiveTransform(DST, SRC)

# ==========================================================
# PARÁMETROS
# ==========================================================

N_WINDOWS = 9
MARGIN    = 50
MINPIX    = 30

EMA_ALPHA = 0.7

left_base_ema  = None
right_base_ema = None

# ==========================================================
# WARP PERSPECTIVE
# ==========================================================

def warp(frame):

    return cv2.warpPerspective(
        frame,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR
    )

# ==========================================================
# THRESHOLD
# ==========================================================

def threshold(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # amarillo
    yellow = cv2.inRange(
        hsv,
        (15, 50, 80),
        (40, 255, 255)
    )

    # blanco
    white = cv2.inRange(
        hsv,
        (0, 0, 180),
        (180, 40, 255)
    )

    combined = cv2.bitwise_or(yellow, white)

    kernel = np.ones((3,3), np.uint8)

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        kernel
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        kernel
    )

    return combined

# ==========================================================
# HISTOGRAMA
# ==========================================================

def histogram_peaks(binary):

    histogram = np.sum(
        binary[binary.shape[0]//2:, :],
        axis=0
    )

    midpoint = histogram.shape[0] // 2

    leftx  = np.argmax(histogram[:midpoint])
    rightx = np.argmax(histogram[midpoint:]) + midpoint

    return leftx, rightx

# ==========================================================
# SLIDING WINDOWS
# ==========================================================

def sliding_windows(binary, leftx_base, rightx_base):

    global left_base_ema
    global right_base_ema

    if left_base_ema is None:
        left_base_ema = leftx_base
    else:
        left_base_ema = (
            EMA_ALPHA * leftx_base +
            (1-EMA_ALPHA) * left_base_ema
        )

    if right_base_ema is None:
        right_base_ema = rightx_base
    else:
        right_base_ema = (
            EMA_ALPHA * rightx_base +
            (1-EMA_ALPHA) * right_base_ema
        )

    leftx_current  = int(left_base_ema)
    rightx_current = int(right_base_ema)

    window_height = binary.shape[0] // N_WINDOWS

    nonzero = binary.nonzero()

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds  = []
    right_lane_inds = []

    out_img = np.dstack((binary, binary, binary))

    for window in range(N_WINDOWS):

        win_y_low  = binary.shape[0] - (window+1)*window_height
        win_y_high = binary.shape[0] - window*window_height

        win_xleft_low  = leftx_current - MARGIN
        win_xleft_high = leftx_current + MARGIN

        win_xright_low  = rightx_current - MARGIN
        win_xright_high = rightx_current + MARGIN

        cv2.rectangle(
            out_img,
            (win_xleft_low, win_y_low),
            (win_xleft_high, win_y_high),
            (0,255,0),
            2
        )

        cv2.rectangle(
            out_img,
            (win_xright_low, win_y_low),
            (win_xright_high, win_y_high),
            (0,255,0),
            2
        )

        good_left_inds = (
            (nonzeroy >= win_y_low) &
            (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &
            (nonzerox < win_xleft_high)
        ).nonzero()[0]

        good_right_inds = (
            (nonzeroy >= win_y_low) &
            (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &
            (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > MINPIX:
            leftx_current = int(
                np.mean(nonzerox[good_left_inds])
            )

        if len(good_right_inds) > MINPIX:
            rightx_current = int(
                np.mean(nonzerox[good_right_inds])
            )

    left_lane_inds  = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return (
        out_img,
        leftx,
        lefty,
        rightx,
        righty
    )

# ==========================================================
# MAIN
# ==========================================================

def detectar_y_analizar_lineas(frame):

    frame = cv2.resize(frame, (640,480))

    warped = warp(frame)

    binary = threshold(warped)

    leftx_base, rightx_base = histogram_peaks(binary)

    (
        debug,
        leftx,
        lefty,
        rightx,
        righty
    ) = sliding_windows(
        binary,
        leftx_base,
        rightx_base
    )

    h, w = binary.shape

    left_detected  = len(leftx)  > 500
    right_detected = len(rightx) > 500

    left_fit  = None
    right_fit = None

    ploty = np.linspace(
        0,
        h-1,
        h
    )

    # ======================================================
    # FIT LEFT
    # ======================================================

    if left_detected:

        left_fit = np.polyfit(
            lefty,
            leftx,
            2
        )

        left_fitx = (
            left_fit[0]*ploty**2 +
            left_fit[1]*ploty +
            left_fit[2]
        )

        left_base = left_fitx[-1]

    else:

        left_fitx = np.ones_like(ploty) * (w*0.3)

        left_base = w * 0.3

    # ======================================================
    # FIT RIGHT
    # ======================================================

    if right_detected:

        right_fit = np.polyfit(
            righty,
            rightx,
            2
        )

        right_fitx = (
            right_fit[0]*ploty**2 +
            right_fit[1]*ploty +
            right_fit[2]
        )

        right_base = right_fitx[-1]

    else:

        right_fitx = np.ones_like(ploty) * (w*0.7)

        right_base = w * 0.7

    # ======================================================
    # CENTRO
    # ======================================================

    lane_center = (
        left_base +
        right_base
    ) / 2.0

    image_center = w / 2.0

    offset = (
        lane_center -
        image_center
    ) / image_center

    offset = np.clip(offset, -1.0, 1.0)

    # ======================================================
    # QUALITY
    # ======================================================

    quality = 0.0

    if left_detected and right_detected:
        quality = 1.0

    elif left_detected or right_detected:
        quality = 0.5

    # ======================================================
    # DIBUJAR CARRILES
    # ======================================================

    lane_vis = np.zeros_like(debug)

    pts_left = np.array([
        np.transpose(
            np.vstack([left_fitx, ploty])
        )
    ])

    pts_right = np.array([
        np.flipud(
            np.transpose(
                np.vstack([right_fitx, ploty])
            )
        )
    ])

    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(
        lane_vis,
        np.int32([pts]),
        (0,255,0)
    )

    # dibujar líneas

    for i in range(len(ploty)-1):

        cv2.line(
            lane_vis,
            (
                int(left_fitx[i]),
                int(ploty[i])
            ),
            (
                int(left_fitx[i+1]),
                int(ploty[i+1])
            ),
            (0,255,255),
            5
        )

        cv2.line(
            lane_vis,
            (
                int(right_fitx[i]),
                int(ploty[i])
            ),
            (
                int(right_fitx[i+1]),
                int(ploty[i+1])
            ),
            (255,255,255),
            5
        )

    # ======================================================
    # UNWARP
    # ======================================================

    unwarped_lane = cv2.warpPerspective(
        lane_vis,
        Mi,
        (640,480)
    )

    result = cv2.addWeighted(
        frame,
        1.0,
        unwarped_lane,
        0.45,
        0
    )

    # ======================================================
    # CENTRO VISUAL
    # ======================================================

    center_x = int(
        lane_center
    )

    cv2.line(
        debug,
        (int(image_center), h),
        (int(image_center), h-80),
        (0,0,255),
        3
    )

    cv2.line(
        debug,
        (center_x, h),
        (center_x, h-80),
        (255,0,0),
        3
    )

    # ======================================================
    # DEBUG TEXT
    # ======================================================

    cv2.putText(
        result,
        f'Offset: {offset:.3f}',
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        result,
        f'Quality: {quality:.2f}',
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    # ======================================================
    # ANALISIS
    # ======================================================

    analisis = {

        "Izquierda": {
            "intermitencia": int(left_detected),
            "x_base": float(left_base),
            "pendiente_prom": 0.0
        },

        "Derecha": {
            "intermitencia": int(right_detected),
            "x_base": float(right_base),
            "pendiente_prom": 0.0
        },

        "Centro": {
            "intermitencia": int(
                left_detected + right_detected
            ),
            "x_base": float(lane_center),
            "pendiente_prom": 0.0
        }

    }

    return result, analisis