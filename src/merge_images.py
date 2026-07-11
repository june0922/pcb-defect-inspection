import os
import glob
import random
import math
import cv2
import numpy as np
from PIL import Image

def apply_color_mapping(binary_img):
    h, w = binary_img.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # 배경색은 녹색 계열만 나오도록 고정
    bg_color = (random.randint(0, 20), random.randint(50, 100), random.randint(0, 20))

    # 회로 색상에서 금색(gold) 제거, 구리색(copper)과 은색(silver)만 남김
    circuit_choice = random.choice(['copper', 'silver'])
    if circuit_choice == 'copper':
        c_color = (51, 115, 184)
    else:
        c_color = (192, 192, 192)

    colored[binary_img == 0] = c_color
    colored[binary_img == 255] = bg_color
    return colored

def add_noise_and_texture(img):
    h, w, c = img.shape
    noise = np.random.normal(0, 15, (h, w, c)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img_noisy

def apply_embossing(colored_img, binary_img):
    edges = cv2.Canny(binary_img, 50, 150)
    h, w = binary_img.shape
    
    shift_h, shift_v = 1, 1
    M_h = np.float32([[1, 0, shift_h], [0, 1, shift_v]])
    M_s = np.float32([[1, 0, -shift_h], [0, 1, -shift_v]])
    
    edges_h = cv2.warpAffine(edges, M_h, (w, h))
    edges_s = cv2.warpAffine(edges, M_s, (w, h))
    
    colored_img = colored_img.astype(np.float32)
    colored_img[edges_h == 255] += 50
    colored_img[edges_s == 255] -= 50
    return np.clip(colored_img, 0, 255).astype(np.uint8)

def apply_blur_and_antialias(img):
    return cv2.GaussianBlur(img, (3, 3), 0.5)

def apply_camera_degradation(img):
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (5, 5), 1.0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 95)]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def apply_augmentation(merged_im):
    img_np = np.array(merged_im.convert('L'))
    _, binary = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
    
    colored = apply_color_mapping(binary)
    textured = add_noise_and_texture(colored)
    embossed = apply_embossing(textured, binary)
    blurred = apply_blur_and_antialias(embossed)
    # 화질 손실을 원천 차단하기 위해 고의적인 카메라 화질 저하(JPEG 시뮬레이션 등) 적용 생략
    final_img = blurred 
    return final_img

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'PCBData'))
    groups = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('group')]
    groups.sort()
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'merged_data'))
    os.makedirs(output_dir, exist_ok=True)
    
    img_width, img_height = 640, 640
    defect_prob = 0.05
    total_created = 0
    
    print("이미지 합성 및 색상 처리 작업을 시작합니다...")
    
    for group in groups:
        group_path = os.path.join(base_dir, group)
        temp_images = glob.glob(os.path.join(group_path, '**', '*_temp.jpg'), recursive=True)
        temp_images.sort(key=lambda x: os.path.basename(x))
        
        available_images = temp_images[:]
        group_created_count = 0
        
        while len(available_images) >= 100:
            S = 10
            needed = 100
            selected_images = available_images[:needed]
            available_images = available_images[needed:]
            
            merged_im = Image.new('RGB', (S * img_width, S * img_height))
            defect_count = 0
            
            for i, temp_path in enumerate(selected_images):
                col = i // S
                row = i % S
                use_defect = random.random() < defect_prob
                img_to_paste = temp_path
                
                if use_defect:
                    test_path = temp_path.replace('_temp.jpg', '_test.jpg')
                    if os.path.exists(test_path):
                        img_to_paste = test_path
                        defect_count += 1
                    else:
                        print(f"경고: 결함 이미지를 찾을 수 없음 ({test_path})")
                
                try:
                    with Image.open(img_to_paste) as im:
                        merged_im.paste(im, (col * img_width, row * img_height))
                except Exception as e:
                    print(f"이미지를 불러오는 중 에러 발생 ({img_to_paste}): {e}")
                    
            group_created_count += 1
            
            # 원본 바이너리 마스크 저장
            base_filename = f"merged_{group}_{S}x{S}_{group_created_count:02d}"
            save_path = os.path.join(output_dir, f"{base_filename}.png")
            merged_im.save(save_path, format="PNG")
            
            # 증강 로직(색상, 노이즈, 엠보싱 등) 적용
            augmented_img = apply_augmentation(merged_im)
            
            # 색상 처리된 이미지도 무손실 PNG로 동일 폴더에 저장
            aug_save_path = os.path.join(output_dir, f"{base_filename}_colored.png")
            cv2.imwrite(aug_save_path, augmented_img)
            
            print(f"[{base_filename}] 마스크(png) 및 색상 이미지(png) 저장 완료 (포함된 결함 이미지 수: {defect_count})")
            total_created += 1

    print(f"\n모든 작업이 완료되었습니다! 총 {total_created}세트의 무손실(png) 합성/색상 이미지가 생성되었습니다.")
    print(f"결과 저장 위치: {output_dir}")

if __name__ == "__main__":
    main()

