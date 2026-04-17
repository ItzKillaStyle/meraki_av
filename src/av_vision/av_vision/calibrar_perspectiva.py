"""
Herramienta de calibración de perspectiva para bird's eye view.
Uso: python3 calibrar_perspectiva.py --video /ruta/al/video.mp4

Instrucciones:
1. El video se pausa en el primer frame
2. Haz clic en 4 puntos que formen un trapecio sobre los carriles
   - Punto 1: esquina inferior izquierda del carril
   - Punto 2: esquina inferior derecha del carril
   - Punto 3: esquina superior derecha del carril (al fondo)
   - Punto 4: esquina superior izquierda del carril (al fondo)
3. Presiona 'c' para confirmar y ver el resultado
4. Presiona 'r' para reiniciar los puntos
5. Presiona 's' para guardar y salir
"""

import cv2
import numpy as np
import argparse
import json

points = []
frame_original = None
frame_display  = None


def click_event(event, x, y, flags, param):
    global points, frame_display

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(frame_display, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(frame_display, str(len(points)),
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(points) > 1:
            cv2.line(frame_display, points[-2], points[-1], (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(frame_display, points[-1], points[0], (0, 255, 0), 2)
            cv2.putText(frame_display,
                        "Presiona 'c' para confirmar, 'r' para reiniciar",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)

        cv2.imshow('Calibracion', frame_display)


def main():
    global points, frame_original, frame_display

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Ruta al video')
    parser.add_argument('--frame', type=int, default=30,
                        help='Frame a usar para calibración (default: 30)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print('Error leyendo el video')
        return

    # Redimensiona a 640x480 — igual que en vision_node
    frame = cv2.resize(frame, (640, 480))
    frame_original = frame.copy()
    frame_display  = frame.copy()
    h, w = frame.shape[:2]

    cv2.namedWindow('Calibracion', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Calibracion', click_event)

    # Dibuja guía
    cv2.putText(frame_display,
                'Clic en 4 puntos del carril (trapecio)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame_display,
                '1:inf-izq  2:inf-der  3:sup-der  4:sup-izq',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    print('\nInstrucciones:')
    print('  Clic izquierdo: agregar punto (4 en total)')
    print('  c: confirmar y ver bird-eye view')
    print('  r: reiniciar puntos')
    print('  s: guardar configuración')
    print('  q: salir sin guardar\n')

    while True:
        cv2.imshow('Calibracion', frame_display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            points = []
            frame_display = frame_original.copy()
            cv2.putText(frame_display,
                        'Clic en 4 puntos del carril (trapecio)',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 255), 2)
            cv2.imshow('Calibracion', frame_display)

        elif key == ord('c') and len(points) == 4:
            # Muestra bird's eye view con los puntos seleccionados
            src = np.float32(points)
            dst = np.float32([
                [w * 0.25, h],
                [w * 0.75, h],
                [w * 0.75, 0],
                [w * 0.25, 0]
            ])
            M    = cv2.getPerspectiveTransform(src, dst)
            warp = cv2.warpPerspective(frame_original, M, (w, h))
            cv2.imshow('Bird Eye View', warp)
            print(f'\nPuntos seleccionados (src):')
            for i, p in enumerate(points):
                print(f'  P{i+1}: {p}')
            print(f'\nDestino (dst):')
            print(f'  [{int(w*0.25)},{h}], [{int(w*0.75)},{h}], '
                  f'[{int(w*0.75)},0], [{int(w*0.25)},0]')

        elif key == ord('s') and len(points) == 4:
            h, w = frame_original.shape[:2]
            config = {
                'perspective_src': points,
                'perspective_dst': [
                    [int(w * 0.25), h],
                    [int(w * 0.75), h],
                    [int(w * 0.75), 0],
                    [int(w * 0.25), 0]
                ],
                'roi_vertices': points,
                'image_size': [w, h]
            }
            with open('perspectiva_calibrada.json', 'w') as f:
                json.dump(config, f, indent=2)
            print('\nGuardado en perspectiva_calibrada.json')
            print('\nAgrega esto a tu vision.yaml:')
            print(f'perspective_src: {points}')
            print(f'perspective_dst: [[{int(w*0.25)},{h}],'
                  f'[{int(w*0.75)},{h}],'
                  f'[{int(w*0.75)},0],'
                  f'[{int(w*0.25)},0]]')
            break

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
