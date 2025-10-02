#!/usr/bin/env python3
"""
YOLOv9t 객체 탐지 예제

이 스크립트는 vision.cpp의 YOLOv9t CLI 기능을 설명하고 사용법을 보여줍니다.
"""

import subprocess
import os
import sys

def run_yolov9t_detection(input_image, output_image, model_path="models/yolov9t_converted-F16.gguf"):
    """
    YOLOv9t를 사용하여 객체 탐지를 실행합니다.
    
    Args:
        input_image: 입력 이미지 경로
        output_image: 출력 이미지 경로  
        model_path: YOLOv9t GGUF 모델 파일 경로
    """
    
    print(f"🔍 YOLOv9t 객체 탐지 실행 중...")
    print(f"   입력 이미지: {input_image}")
    print(f"   모델: {model_path}")
    print(f"   출력: {output_image}")
    print()
    
    # vision-cli yolov9t 명령 실행
    cmd = [
        "./build/bin/vision-cli", "yolov9t",
        "-m", model_path,
        "-i", input_image,
        "-o", output_image
    ]
    
    try:
        result = subprocess.run(cmd, cwd=".", check=True, capture_output=True, text=True)
        print("✅ 객체 탐지 완료!")
        print()
        print("=== CLI 출력 ===")
        print(result.stdout)
        if result.stderr:
            print("=== 경고/오류 ===")
            print(result.stderr)
        print()
        
        if os.path.exists(output_image):
            file_size = os.path.getsize(output_image)
            print(f"📸 출력 이미지 생성됨: {output_image} ({file_size:,} bytes)")
        else:
            print("❌ 출력 이미지가 생성되지 않았습니다.")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류 발생: {e}")
        print("=== 오류 출력 ===")
        print(e.stdout)
        print(e.stderr)
        return False
    
    return True

def main():
    """메인 함수 - 여러 이미지로 YOLOv9t 테스트"""
    
    print("🎯 YOLOv9t 객체 탐지 예제")
    print("=" * 50)
    print()
    
    # 입력 이미지들
    test_images = [
        "tests/input/cat-and-hat.jpg",
        "tests/input/bench-image.jpg", 
        "tests/input/vase-and-bowl.jpg",
    ]
    
    # 모델 파일들
    models = [
        "models/yolov9t_converted-F16.gguf",  # FP16 버전
        "models/yolov9t_converted.gguf",      # FP32 버전
    ]
    
    results_dir = "tests/results"
    os.makedirs(results_dir, exist_ok=True)
    
    success_count = 0
    total_tests = 0
    
    for model in models:
        if not os.path.exists(model):
            print(f"⚠️  모델 파일을 찾을 수 없습니다: {model}")
            continue
            
        model_name = os.path.basename(model).replace('.gguf', '')
        print(f"🤖 모델: {model_name}")
        print("-" * 30)
        
        for input_image in test_images:
            if not os.path.exists(input_image):
                print(f"⚠️  입력 이미지를 찾을 수 없습니다: {input_image}")
                continue
                
            image_name = os.path.basename(input_image).split('.')[0]
            output_image = f"{results_dir}/yolov9t_{model_name}_{image_name}.png"
            
            total_tests += 1
            if run_yolov9t_detection(input_image, output_image, model):
                success_count += 1
            print()
    
    print("=" * 50)
    print(f"📊 테스트 결과: {success_count}/{total_tests} 성공")
    
    if success_count > 0:
        print()
        print("🎉 YOLOv9t CLI가 성공적으로 작동합니다!")
        print()
        print("📁 생성된 파일들:")
        for file in os.listdir(results_dir):
            if file.startswith("yolov9t_"):
                full_path = os.path.join(results_dir, file)
                size = os.path.getsize(full_path)
                print(f"   {file} ({size:,} bytes)")
        print()
        print("💡 현재 구현 상태:")
        print("   ✅ CLI 인터페이스 완성")
        print("   ✅ 모델 로딩 및 텐서 접근")
        print("   ✅ 이미지 입출력 처리")
        print("   🔄 완전한 추론 파이프라인 (개발 진행 중)")
        print()
        print("🚀 다음 단계:")
        print("   1. 실제 YOLOv9t 순방향 패스 구현")
        print("   2. 객체 탐지 박스 그리기")
        print("   3. NMS (Non-Maximum Suppression) 후처리")
        print("   4. COCO 클래스 레이블 출력")

if __name__ == "__main__":
    main()