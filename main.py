import os
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from database import conectar_db


# ===========================
# Ruta del modelo entrenado para bolsas
MODEL_PATH = r"C:\Users\fbver\Documents\Ocean_ai\modelo_bolsas\models\plastico_detector_best .pt"

# Verificación simple
if not os.path.exists(MODEL_PATH):
    print(f"❌ No se encontró: {MODEL_PATH}")
    print("📁 Archivos disponibles:")
    carpeta = r"C:\Users\fbver\Documents\ocean_ai\modelo_bolsas\models"
    for archivo in os.listdir(carpeta):
        if '.pt' in archivo:
            print(f"   - {archivo}")
    raise FileNotFoundError(f"❌ Modelo no encontrado. Verifica el nombre del archivo.")

print(f"🔹 Cargando modelo desde: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("✅ Modelo de bolsas plásticas cargado correctamente\n")
# Clase para bolsas plásticas (ajusta según tu modelo entrenado)
# Normalmente será la clase 0 si solo tienes una clase
BAG_CLASS_ID = 0

# ===========================
# Conexión a la base de datos
conexion = conectar_db()
cursor = conexion.cursor() if conexion else None

# ===========================
# Configurar cámara
cap = cv2.VideoCapture(0)  # 0 = cámara principal del laptop
if not cap.isOpened():
    raise IOError("❌ No se pudo abrir la cámara")

zona = "A"  # debe existir como zona en la tabla Camara
print(f"🎥 Iniciando detección de bolsas en zona {zona} (presiona 'q' para salir)\n")

# ===========================
# Variables de control
ultimo_update = time.time()
ids_vivos = {}  # track_id : último_timestamp_visto
TIEMPO_MAX_INACTIVIDAD = 2.0  # segundos antes de considerar que se fue

# ===========================
# Bucle de detección y tracking
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame")
        break

    results = model.track(
        frame,
        tracker="tracker_config.yaml", 
        persist=True,
        verbose=False
    )

    tiempo_actual = time.time()
    ids_visibles = set()

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None

            # Detectar solo bolsas plásticas
            if cls == BAG_CLASS_ID and conf >= 0.1 and track_id is not None:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Dibujar bounding box
                color = (0, 255, 0)  # Verde para bolsas
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Bolsa ID:{track_id} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                ids_visibles.add(track_id)
                ids_vivos[track_id] = tiempo_actual

                if cursor:
                    try:
                        cursor.execute("SELECT 1 FROM Objeto WHERE id = %s", (track_id,))
                        existe = cursor.fetchone()
                        if not existe:
                            cursor.execute("""
                                INSERT INTO Objeto (id, zona, categoria, fecha_hora, confianza, imagen_url)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            """, (track_id, zona, "bolsa_plastica", datetime.now(), conf, f"/imagenes/{track_id}.jpg"))
                            conexion.commit()
                            print(f"🟢 Nueva bolsa detectada → ID={track_id} | Confianza: {conf:.2f}")
                    except Exception as e:
                        print("❌ Error insertando en Objeto:", e)
                        conexion.rollback()

    # Eliminar IDs que no se han visto por un tiempo
    ids_vivos = {
        i: t for i, t in ids_vivos.items()
        if tiempo_actual - t < TIEMPO_MAX_INACTIVIDAD
    }

    # Actualizar conteo en la base de datos cada 1 segundo
    if tiempo_actual - ultimo_update >= 1:
        try:
            conteo_actual = len(ids_vivos)
            if cursor:
                cursor.execute("""
                    UPDATE Camara
                    SET conteo_bolsas = %s
                    WHERE zona = %s
                """, (conteo_actual, zona))
                conexion.commit()
            print(f"📊 [{datetime.now().strftime('%H:%M:%S')}] Zona {zona} → {conteo_actual} bolsas detectadas")
            ultimo_update = tiempo_actual
        except Exception as e:
            print("❌ Error actualizando conteo:", e)
            conexion.rollback()

    # Mostrar ventana con detección
    cv2.putText(frame, f"Bolsas detectadas: {len(ids_vivos)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    
    # Mostrar información del modelo
    cv2.putText(frame, "Modelo: Deteccion de Bolsas Plasticas", 
                (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Detección de Bolsas Plásticas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===========================
# Cierre de recursos
cap.release()
cv2.destroyAllWindows()
if cursor:
    cursor.close()
if conexion:
    conexion.close()

print("🟦 Sistema de detección de bolsas finalizado correctamente.")