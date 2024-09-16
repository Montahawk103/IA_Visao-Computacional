import cv2
import numpy as np
from datetime import timedelta
from collections import deque

class BurgerCounter:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.burger_count = 0
        self.empty_basket_count = 0
        self.filled_basket_count = 0
        self.basket_entry_times = []
        self.basket_exit_times = []
        self.burger_times = []
        
        # Obter dimensões do vídeo
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Definir regiões de interesse (ROI)
        basket_entry_height = int(self.frame_height * 0.1720)
        self.basket_entry_roi = (0, 0, self.frame_width, basket_entry_height)
        self.basket_exit_roi = (0, self.frame_height-100, self.frame_width, 100)
        burger_roi_width = int(self.frame_width * 0.3)
        self.burger_roi = (self.frame_width - burger_roi_width, 0, burger_roi_width, self.frame_height)
        
        # Parâmetros para detecção
        self.min_basket_area = 1000
        self.max_basket_area = 10000
        self.min_burger_area = 100
        self.max_burger_area = 500
        self.burgers_per_basket = 4

        # Buffers para suavizar detecções
        self.basket_entry_buffer = deque(maxlen=3)
        self.basket_exit_buffer = deque(maxlen=3)
        self.burger_buffer = deque(maxlen=3)

        # Tempos da última detecção
        self.last_basket_entry_time = 0
        self.last_basket_exit_time = 0
        self.last_burger_time = 0

    def process_video(self):
        frame_count = 0
        fps = self.video.get(cv2.CAP_PROP_FPS)
        
        basket_entry_backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
        basket_exit_backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
        burger_backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)

        while True:
            ret, frame = self.video.read()
            if not ret or frame is None:
                break
            
            frame_count += 1
            current_time = frame_count / fps

            self.process_basket_entry(frame, basket_entry_backSub, current_time)
            self.process_basket_exit(frame, basket_exit_backSub, current_time)
            self.process_burgers(frame, burger_backSub, current_time)

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
                self.empty_basket_count += 1
                self.basket_entry_times.append(current_time)
                self.last_basket_entry_time = current_time
                print(f"Nova cesta vazia {self.empty_basket_count} detectada no tempo {current_time:.2f}s")
                print(f"Total de cestas vazias detectadas: {len(self.basket_entry_times)}")

    def process_basket_exit(self, frame, backSub, current_time):
        roi = self.get_safe_roi(frame, self.basket_exit_roi)
        if roi is not None:
            fgMask = backSub.apply(roi)
            _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            basket_detected = any(self.min_basket_area < cv2.contourArea(contour) < self.max_basket_area for contour in contours)
            self.basket_exit_buffer.append(basket_detected)

            if all(self.basket_exit_buffer) and current_time - self.last_basket_exit_time > 2.0:
                self.filled_basket_count += 1
                self.burger_count += self.burgers_per_basket
                self.basket_exit_times.append(current_time)
                self.last_basket_exit_time = current_time
                print(f"Cesta cheia {self.filled_basket_count} saiu da tela no tempo {current_time:.2f}s")
                print(f"Total de cestas cheias detectadas: {len(self.basket_exit_times)}")

    def process_burgers(self, frame, backSub, current_time):
        roi = self.get_safe_roi(frame, self.burger_roi)
        if roi is not None:
            fgMask = backSub.apply(roi)
            _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            burger_detected = any(self.min_burger_area < cv2.contourArea(contour) < self.max_burger_area for contour in contours)
            self.burger_buffer.append(burger_detected)

            if all(self.burger_buffer) and current_time - self.last_burger_time > 0.5:
                self.burger_times.append(current_time)
                self.last_burger_time = current_time
                print(f"Hambúrguer detectado no tempo {current_time:.2f}s")

    def get_safe_roi(self, frame, roi):
        x, y, w, h = roi
        if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
            return frame[y:y+h, x:x+w]
        return None

    def draw_visualization(self, frame):
        vis_frame = frame.copy()
        
        cv2.rectangle(vis_frame, (self.basket_entry_roi[0], self.basket_entry_roi[1]), 
                      (self.basket_entry_roi[0]+self.basket_entry_roi[2], self.basket_entry_roi[1]+self.basket_entry_roi[3]), (255, 0, 0), 2)
        cv2.rectangle(vis_frame, (self.basket_exit_roi[0], self.basket_exit_roi[1]), 
                      (self.basket_exit_roi[0]+self.basket_exit_roi[2], self.basket_exit_roi[1]+self.basket_exit_roi[3]), (0, 0, 255), 2)
        cv2.rectangle(vis_frame, (self.burger_roi[0], self.burger_roi[1]), 
                      (self.burger_roi[0]+self.burger_roi[2], self.burger_roi[1]+self.burger_roi[3]), (0, 255, 0), 2)
        
        cv2.putText(vis_frame, f'Burgers: {self.burger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_frame, f'Empty Baskets: {self.empty_basket_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(vis_frame, f'Filled Baskets: {self.filled_basket_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return vis_frame

    def calculate_statistics(self):
        if self.basket_exit_times and self.basket_entry_times:
            min_times = min(len(self.basket_exit_times), len(self.basket_entry_times))
            exit_times = np.array(self.basket_exit_times[:min_times])
            entry_times = np.array(self.basket_entry_times[:min_times])
            
            time_diffs = exit_times - entry_times
            valid_diffs = time_diffs[time_diffs > 0]
            
            if len(valid_diffs) > 0:
                self.avg_basket_fill_time = np.mean(valid_diffs)
            else:
                self.avg_basket_fill_time = 0
        else:
            self.avg_basket_fill_time = 0

        self.avg_burger_time = np.mean(np.diff(self.burger_times)) if len(self.burger_times) > 1 else 0

    def generate_logs(self):
        print(f"Total de hambúrgueres: {self.burger_count}")
        print(f"Total de cestas vazias: {self.empty_basket_count}")
        print(f"Total de cestas cheias: {self.filled_basket_count}")
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