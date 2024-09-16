import cv2
import numpy as np
from datetime import timedelta
from collections import deque

class BurgerCounter:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.burger_count = 0
        self.basket_count = 0
        self.basket_entry_times = []
        self.basket_exit_times = []
        
        # Obter dimensões do vídeo
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Definir regiões de interesse (ROI)
        self.basket_entry_roi = (int(self.frame_width*0.3), 50, int(self.frame_width*0.4), 100)  # Entrada de cestas vazias
        self.basket_exit_roi = (0, self.frame_height-100, self.frame_width, 100)  # Saída de cestas cheias
        
        # Parâmetros para detecção
        self.min_basket_area = 1000
        self.max_basket_area = 10000
        self.burgers_per_basket = 4

        # Buffers para suavizar detecções
        self.basket_entry_buffer = deque(maxlen=3)
        self.basket_exit_buffer = deque(maxlen=3)

        # Tempos da última detecção
        self.last_basket_entry_time = 0
        self.last_basket_exit_time = 0

    def process_video(self):
        frame_count = 0
        fps = self.video.get(cv2.CAP_PROP_FPS)
        
        basket_entry_backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
        basket_exit_backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)

        while True:
            ret, frame = self.video.read()
            if not ret or frame is None:
                break
            
            frame_count += 1
            current_time = frame_count / fps

            # Processar ROIs
            self.process_basket_entry(frame, basket_entry_backSub, current_time)
            self.process_basket_exit(frame, basket_exit_backSub, current_time)

            # Visualização
            vis_frame = self.draw_visualization(frame)
            cv2.imshow('Burger Counter Visualization', vis_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()
        self.calculate_statistics()

    def process_basket_entry(self, frame, backSub, current_time):
        roi = self.get_safe_roi(frame, self.basket_entry_roi)
        if roi is not None:
            fgMask = backSub.apply(roi)
            _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            basket_detected = any(self.min_basket_area < cv2.contourArea(contour) < self.max_basket_area for contour in contours)
            self.basket_entry_buffer.append(basket_detected)

            if all(self.basket_entry_buffer) and current_time - self.last_basket_entry_time > 2.0:
                self.basket_entry_times.append(current_time)
                self.last_basket_entry_time = current_time
                print(f"Nova cesta vazia detectada no tempo {current_time:.2f}s")

    def process_basket_exit(self, frame, backSub, current_time):
        roi = self.get_safe_roi(frame, self.basket_exit_roi)
        if roi is not None:
            fgMask = backSub.apply(roi)
            _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            basket_detected = any(self.min_basket_area < cv2.contourArea(contour) < self.max_basket_area for contour in contours)
            self.basket_exit_buffer.append(basket_detected)

            if all(self.basket_exit_buffer) and current_time - self.last_basket_exit_time > 2.0:
                self.basket_count += 1
                self.burger_count += self.burgers_per_basket
                self.basket_exit_times.append(current_time)
                self.last_basket_exit_time = current_time
                print(f"Cesta {self.basket_count} cheia saiu da tela no tempo {current_time:.2f}s com {self.burgers_per_basket} hambúrgueres")

    def get_safe_roi(self, frame, roi):
        x, y, w, h = roi
        if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
            return frame[y:y+h, x:x+w]
        return None

    def draw_visualization(self, frame):
        vis_frame = frame.copy()
        
        # Desenhar ROIs
        cv2.rectangle(vis_frame, (self.basket_entry_roi[0], self.basket_entry_roi[1]), 
                      (self.basket_entry_roi[0]+self.basket_entry_roi[2], self.basket_entry_roi[1]+self.basket_entry_roi[3]), (255, 0, 0), 2)
        cv2.rectangle(vis_frame, (self.basket_exit_roi[0], self.basket_exit_roi[1]), 
                      (self.basket_exit_roi[0]+self.basket_exit_roi[2], self.basket_exit_roi[1]+self.basket_exit_roi[3]), (0, 0, 255), 2)
        
        # Adicionar informações na tela
        cv2.putText(vis_frame, f'Burgers: {self.burger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_frame, f'Baskets: {self.basket_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return vis_frame

    def calculate_statistics(self):
        self.avg_basket_fill_time = np.mean(np.array(self.basket_exit_times) - np.array(self.basket_entry_times[:len(self.basket_exit_times)])) if self.basket_exit_times else 0
        self.avg_burger_time = self.avg_basket_fill_time / self.burgers_per_basket if self.avg_basket_fill_time > 0 else 0

    def generate_logs(self):
        print(f"Total de hambúrgueres: {self.burger_count}")
        print(f"Total de cestas: {self.basket_count}")
        print(f"Tempos de entrada das cestas vazias: {[self.format_time(t) for t in self.basket_entry_times]}")
        print(f"Tempos de saída das cestas cheias: {[self.format_time(t) for t in self.basket_exit_times]}")
        print(f"Tempo médio de preenchimento de cesta: {self.format_time(self.avg_basket_fill_time)}")
        print(f"Tempo médio de passagem de hambúrguer: {self.format_time(self.avg_burger_time)}")

    @staticmethod
    def format_time(seconds):
        return str(timedelta(seconds=seconds))

if __name__ == "__main__":
    counter = BurgerCounter('burger-trim.mp4')
    counter.process_video()
    counter.generate_logs()