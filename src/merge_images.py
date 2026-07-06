import os
import glob
import random
from PIL import Image

def get_image_paths():
    # dataset 폴더 경로 (스크립트 위치 기준 상위 폴더의 dataset/PCBData)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'PCBData'))
    test_images = glob.glob(os.path.join(base_dir, '**', '*_test.jpg'), recursive=True)
    
    images_info = []
    for test_img in test_images:
        directory = os.path.dirname(test_img)
        basename = os.path.basename(test_img)
        prefix = basename.replace('_test.jpg', '')
        
        test_path = test_img
        temp_path = os.path.join(directory, f"{prefix}_temp.jpg")
        
        if os.path.exists(temp_path):
            images_info.append({
                'test': test_path,
                'temp': temp_path,
                'id': prefix
            })
            
    return images_info

def main():
    print("이미지 경로를 검색하는 중입니다...")
    images_info = get_image_paths()
    print(f"총 {len(images_info)} 쌍의 이미지를 찾았습니다.")
    
    if len(images_info) != 1500:
        print(f"경고: 1500장의 이미지를 예상했으나 {len(images_info)}장이 발견되었습니다.")
        if len(images_info) == 0:
            print("이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
            return

    # 1500장의 이미지를 중복 없이 무작위로 사용하기 위해 셔플 (랜덤성 부여)
    random.shuffle(images_info)

    # 16가지의 그리드 조합 (행, 열)
    combinations = [
        (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 2), (4, 3), (4, 4)
    ]
    
    # 1500장의 이미지를 모두 소진하기 위해 위 16가지 조합을 각각 15번씩 반복 수행
    # (총 필요 이미지 장수 = 100장 * 15세트 = 1500장)
    sets_count = 15
    tasks = combinations * sets_count
    
    # 저장될 폴더 경로
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'merged_data'))
    os.makedirs(output_dir, exist_ok=True)
    
    image_idx = 0
    img_width, img_height = 640, 640
    defect_prob = 0.0625  # 6.25% 확률로 결함 이미지 사용
    
    # 조합별 생성 횟수를 카운트하기 위한 딕셔너리
    counts = {f"{r}x{c}": 0 for r, c in combinations}

    print("이미지 합성 작업을 시작합니다...")
    
    for (r, c) in tasks:
        needed = r * c
        if image_idx + needed > len(images_info):
            print(f"남은 이미지가 부족하여 {r}x{c} 합성을 중단합니다.")
            break
            
        group = images_info[image_idx : image_idx + needed]
        image_idx += needed
        
        # 배경이 될 빈 캔버스 이미지 생성
        merged_im = Image.new('RGB', (c * img_width, r * img_height))
        
        defect_count = 0
        for i, info in enumerate(group):
            row = i // c
            col = i % c
            
            # 지정된 확률(6.25%)에 따라 결함 이미지 사용 여부 결정
            if random.random() < defect_prob:
                img_path = info['test']  # 결함 이미지
                defect_count += 1
            else:
                img_path = info['temp']  # 정상 이미지
                
            try:
                with Image.open(img_path) as im:
                    merged_im.paste(im, (col * img_width, row * img_height))
            except Exception as e:
                print(f"이미지를 불러오는 중 에러 발생 ({img_path}): {e}")
                
        # 생성된 캔버스 저장
        counts[f"{r}x{c}"] += 1
        idx = counts[f"{r}x{c}"]
        filename = f"merged_{r}x{c}_{idx:02d}.jpg"
        save_path = os.path.join(output_dir, filename)
        
        merged_im.save(save_path, quality=95)
        print(f"[{filename} 저장 완료] (포함된 결함 이미지 수: {defect_count})")

    total_created = sum(counts.values())
    print(f"\n모든 작업이 완료되었습니다! 총 {total_created}장의 합성 이미지가 생성되었습니다.")
    print(f"결과 저장 위치: {output_dir}")

if __name__ == "__main__":
    main()
